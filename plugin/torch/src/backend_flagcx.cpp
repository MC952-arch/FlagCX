#include "backend_flagcx.hpp"
#include <iostream>

namespace c10d
{
    namespace {
        // FlagCX op mapping
        const std::map<ReduceOp::RedOpType, flagcxRedOp_t> flagcxOp = {
            {ReduceOp::MIN, flagcxMin},
            {ReduceOp::MAX, flagcxMax},
            {ReduceOp::SUM, flagcxSum},
            {ReduceOp::PRODUCT, flagcxProd},
            {ReduceOp::AVG, flagcxAvg},
        };

        flagcxRedOp_t getFlagcxReduceOp(
            const ReduceOp &reduceOp,
            at::Tensor &input,
            const flagcxDataType_t &dataType,
            const flagcxComm_t &comm)
        {
            try
            {
                if (input.scalar_type() == at::kBool)
                {
                    if (reduceOp == ReduceOp::SUM)
                    {
                        // For bool tensors, map sum to max, which both represent a bitwise or.
                        // This is to prevent overflow issues with sum, since we use uint8 to
                        // represent a bool (see ncclDataType mapping).
                        return flagcxMax;
                    }
                    if (reduceOp == ReduceOp::AVG)
                    {
                        C10_THROW_ERROR(
                            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
                    }
                }
                return flagcxOp.at(reduceOp);
            }
            catch (const std::out_of_range &)
            {
                switch (reduceOp)
                {
                case ReduceOp::AVG:
                    C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.AVG with FlagCX");
                    break;
                case ReduceOp::BAND:
                    C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with FlagCX");
                    break;
                case ReduceOp::BOR:
                    C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with FlagCX");
                    break;
                case ReduceOp::BXOR:
                    C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with FlagCX");
                    break;
                default:
                    C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
                    break;
                }
            }
        }

        // FlagCX type typing
        std::map<at::ScalarType, flagcxDataType_t> flagcxDataType = {
            {at::kChar, flagcxInt8},
            {at::kByte, flagcxUint8},
            {at::kFloat, flagcxFloat},
            {at::kDouble, flagcxDouble},
            {at::kInt, flagcxInt32},
            {at::kLong, flagcxInt64},
            {at::kHalf, flagcxHalf},
            {at::kBool, flagcxUint8},
            {at::kFloat8_e5m2, flagcxUint8},
            {at::kFloat8_e4m3fn, flagcxUint8},
            {at::kFloat8_e4m3fnuz, flagcxUint8},
            {at::kFloat8_e5m2fnuz, flagcxUint8},
            {at::kBFloat16, flagcxBfloat16},
        };

        // Helper function that gets the data type and issues error if not supported
        flagcxDataType_t getFlagcxDataType(at::ScalarType type)
        {
            auto it = flagcxDataType.find(type);
            TORCH_CHECK_WITH(
                TypeError,
                it != flagcxDataType.end(),
                "Input tensor data type is not supported for FlagCX process group: ",
                type);
            return it->second;
        }

        bool check_same_size(const std::vector<at::Tensor>& input_tensors)
        {
            for (const auto &input_tensor : input_tensors)
            {
                if (!input_tensors[0].is_same_size(input_tensor))
                {
                    return false;
                }
            }
            return true;
        }

        void check_device(at::Device dev1, at::Device dev2)
        {
#ifdef USE_CAMBRICON_ADAPTOR
            if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2)
            {
                throw std::runtime_error("BackendFlagcx does not support multidevice tensors");
            }
#else
            if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2)
            {
                throw std::runtime_error("BackendFlagcx does not support multidevice tensors");
            }
#endif
        }
    } // namespace

    bool WorkFlagcx::isCompleted()
    {
        return future_->completed();
    }

    bool WorkFlagcx::isSuccess() const
    {
        return future_->hasValue();
    }

    bool WorkFlagcx::wait(std::chrono::milliseconds /* unused */)
    {
        future_->wait();
        return true;
    }

    c10::intrusive_ptr<c10::ivalue::Future> WorkFlagcx::getFuture()
    {
        return future_;
    }

    // If necessary, pass store/rank/size to the ctor and exchange connection
    // information here
    BackendFlagcx::BackendFlagcx(
        const c10::intrusive_ptr<::c10d::Store>& store,
        int rank, int size)
        : Backend(rank, size), store(store)
    {
        flagcxHandleInit(&handler);
        handler->devHandle->getDeviceCount(&nDevs);
#ifdef USE_NVIDIA_ADAPTOR
        event = std::make_unique<CUDAEventFlagcx>();
#elif USE_ILUVATAR_COREX_ADAPTOR
        event = std::make_unique<IXCUDAEventFlagcx>();
#elif USE_CAMBRICON_ADAPTOR
        event = std::make_unique<MLUEventFlagcx>();
#endif
    }

    BackendFlagcx::~BackendFlagcx()
    {
        if (*(handler->status) == 1)
        {
            handler->devHandle->streamDestroy(stream);
            flagcxCommDestroy(handler->comm);
            *(handler->status) = 0;
        }
        if (*(handler->status) == 0)
        {
            flagcxHandleFree(handler);
        }
    }

    void BackendFlagcx::initComm(at::Device dev)
    {
        if (*(handler->status) == 0)
        {
            // Get the unique id
            flagcxGetUniqueId(&handler->uniqueId);
            if (rank_ == 0)
            {
                auto vec = std::vector<uint8_t>(
                    reinterpret_cast<uint8_t *>(handler->uniqueId),
                    reinterpret_cast<uint8_t *>(handler->uniqueId) + sizeof(flagcxUniqueId));
                store->set("flagcx/unique_id", std::string(vec.begin(), vec.end()));
            }
            else
            {
                try
                {
                    auto vec = store->get("flagcx/unique_id");
                    TORCH_CHECK_WITH(
                        DistBackendError,
                        vec.size() == sizeof(flagcxUniqueId),
                        "Invalide size for flagcxUniqueId");
                    std::memcpy((uint8_t *)handler->uniqueId, vec.data(), sizeof(flagcxUniqueId));
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error(
                        "Failed to retrieve the unique id from the store: " +
                        std::string(e.what()));
                }
                catch (...)
                {
                    throw std::runtime_error(
                        "Unknown exception during the retrieving of unique id from the store");
                }
            }
            // Initialize the communicator
            flagcxCommInitRank(&handler->comm, size_, handler->uniqueId, rank_);
            // Initialize the stream
            handler->devHandle->streamCreate(&stream);
            *(handler->status) = 1;
        }
        else
        {
            if (dev.is_cuda() || dev.is_privateuseone())
            {
                // Get the device index
                int device_id;
                flagcxCommGetDeviceNumber(handler->comm, &device_id);
                if (device_id != dev.index())
                {
                    throw std::runtime_error(
                        "flagcx communicator was initialized with different device");
                }
            }
        }
    }

    void BackendFlagcx::syncStream(at::Device device)
    {
        event->record(device);
        event->block(stream, device);
    }

    c10::intrusive_ptr<Work> BackendFlagcx::allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& /* unused */)
    {
        auto inputTensor = inputTensors.back();
        auto outputTensors_ = outputTensors.back();
        auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
        check_device(inputTensor.device(), outputTensors_[0].device());
        initComm(inputTensor.device());
        syncStream(inputTensor.device());

        if (!check_same_size(outputTensors_))
        {
            throw std::runtime_error("flagcx only support same size allgather operation");
        }
        else
        {
            // Flatten a vector of tensors into a single, stacked tensor.
            at::Tensor outputFlattened = newLikeFlat(outputTensors_);

            // Sync the underlying stream with flagcx stream.
            syncStream(inputTensor.device());

            // Perform the allgather operation
            flagcxAllGather(
                inputTensor.data_ptr(),
                outputFlattened.data_ptr(),
                inputTensor.numel(),
                flagcxDataType,
                handler->comm,
                stream);

            // Unflatten the flattened tensor back into a vector of tensors.
            for (const auto j : c10::irange(outputTensors_.size()))
            {
                outputTensors_[j].copy_(outputFlattened[j], true);
            }
        }

        // Create a future to track the allgather operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensors_));
        return c10::make_intrusive<WorkFlagcx>(OpType::ALLGATHER, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::_allgather_base(
        at::Tensor& /* unused */,
        at::Tensor& /* unused */,
        const AllgatherOptions& /* unused */)
    {
        throw std::runtime_error("BackendFlagcx does not support _allgather_base");
    }

    c10::intrusive_ptr<Work> BackendFlagcx::allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts)
    {
        auto &tensor = tensors.back();
        auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
        auto flagcxReduceOp = getFlagcxReduceOp(opts.reduceOp, tensor, flagcxDataType, handler->comm);
        initComm(tensor.device());
        syncStream(tensor.device());

        // Perform the allreduce operation
        flagcxAllReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            flagcxDataType,
            flagcxReduceOp,
            handler->comm,
            stream);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::ALLREDUCE, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::allreduce_coalesced(
        std::vector<at::Tensor>& /* unused */,
        const AllreduceCoalescedOptions& /* unused */)
    {
        throw std::runtime_error("BackendFlagcx does not support allreduce_coalesced");
    }

    c10::intrusive_ptr<Work> BackendFlagcx::alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllToAllOptions& /* unused */)
    {
        // std::vector<int64_t> inSplitSizes;
        // std::vector<int64_t> outSplitSizes;
        // for (const auto r : c10::irange(outputTensors.size()))
        // {
        //     inSplitSizes.push_back(inputTensors[r].numel());
        //     outSplitSizes.push_back(outputTensors[r].numel());
        // }

        if (inputTensors.size() != outputTensors.size())
            throw std::runtime_error("Input and output tensors size must be equal");
        if (inputTensors[0].numel() != outputTensors[0].numel())
            throw std::runtime_error("Input and output tensors must have the same number of elements");
        if (!check_same_size(inputTensors) || !check_same_size(outputTensors))
            throw std::runtime_error("flagcx only support same size alltoall operation");
        if (getFlagcxDataType(inputTensors[0].scalar_type()) != getFlagcxDataType(outputTensors[0].scalar_type()))
            throw std::runtime_error("Input and output tensors must have the same data type");
        auto count = inputTensors[0].numel();
        auto flagcxDataType = getFlagcxDataType(inputTensors[0].scalar_type());

        auto device = outputTensors[0].device();
        initComm(device);
        syncStream(device);

        // Flatten a vector of tensors into a single, stacked tensor.
        // TODO: check the correctness of newLikeFlat
        // if (rank_ == 7)
        // {
        //     for (const auto j : c10::irange(inputTensors.size()))
        //     {
        //         std::cout << "inputTensors[" << j << "]: " << inputTensors[j] << "; outputTensors[" << j << "]: " << outputTensors[j] << std::endl;
        //     }
        // }
        at::Tensor inputFlattened = newLikeFlat(inputTensors);
        at::Tensor outputFlattened = newLikeFlat(outputTensors);
        // if (rank_ == 7)
        // {
        //     std::cout << "inputFlattened: " << inputFlattened << "; outputFlattened: " << outputFlattened << std::endl;
        // }

        flagcxAlltoAll(
            inputFlattened.data_ptr(),
            outputFlattened.data_ptr(),
            count,
            flagcxDataType,
            handler->comm,
            stream);

        // Unflatten the flattened tensor back into a vector of tensors.
        for (const auto j : c10::irange(outputTensors.size()))
        {
            inputTensors[j].copy_(inputFlattened[j], true);
            outputTensors[j].copy_(outputFlattened[j], true);
        }

        // Create a future to track the alltoall operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::ALLTOALL, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const AllToAllOptions& /* unused */)
    {
        throw std::runtime_error("BackendFlagcx does not support alltoall_base");
    }

    c10::intrusive_ptr<Work> BackendFlagcx::barrier(
        const BarrierOptions& opts)
    {
        throw std::runtime_error("BackendFlagcx does not support barrier");
        // int barDevIdx = static_cast<int16_t>(rank_ % localDeviceCount_);
        // auto barDevice = at::Device(at::DeviceType::CUDA, static_cast<c10::DeviceIndex>(barDevIdx));

        // // Create a dummy tensor on the device
        // // Note: we use zeros() instead of empty() to prevent barrier from triggering
        // // alarm when NaN checker is enabled.
        // std::vector<at::Tensor> barrierTensors;
        // barrierTensors.emplace_back(at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat)));

        // // All reduce to achieve the barrier
        // auto work = allreduce(barrierTensors);

        // // Work will take over barrierTensors
        // auto flagcxWork = dynamic_cast<WorkFlagcx *>(work.get());
        // TORCH_CHECK(flagcxWork);
        // flagcxWork->isBarrierOp_ = true;
        // return work;
    }

    c10::intrusive_ptr<Work> BackendFlagcx::broadcast(
        std::vector<at::Tensor>& tensors,
        const BroadcastOptions& opts)
    {
        auto &tensor = tensors.back();
        auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
        initComm(tensor.device());
        syncStream(tensor.device());

        const auto root = opts.rootRank + opts.rootTensor;

        flagcxBroadcast(
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            flagcxDataType,
            root,
            handler->comm,
            stream);

        // Create a future to track the broadcast operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::BROADCAST, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::gather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const GatherOptions& opts)
    {
        auto& inputTensor = inputTensors.back();
        auto& outputTensors_ = outputTensors.back();
        auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
        initComm(inputTensor.device());
        syncStream(inputTensor.device());

        auto root = opts.rootRank;
        // Flatten a vector of tensors into a single, stacked tensor.
        at::Tensor outputFlattened = newLikeFlat(outputTensors_);

        // Perform the gather operation
        flagcxGather(
            inputTensor.data_ptr(),
            outputFlattened.data_ptr(),
            inputTensor.numel(),
            flagcxDataType,
            root,
            handler->comm,
            stream);

        // Unflatten the flattened tensor back into a vector of tensors.
        for (const auto j : c10::irange(outputTensors_.size()))
        {
            outputTensors_[j].copy_(outputFlattened[j], true);
        }

        // Create a future to track the gather operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensors_));
        return c10::make_intrusive<WorkFlagcx>(OpType::GATHER, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::reduce(
        std::vector<at::Tensor>& tensors,
        const ReduceOptions& opts)
    {
        auto &tensor = tensors.back();
        auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
        auto flagcxReduceOp = getFlagcxReduceOp(opts.reduceOp, tensor, flagcxDataType, handler->comm);
        initComm(tensor.device());
        syncStream(tensor.device());

        const auto root = opts.rootRank + opts.rootTensor;

        flagcxReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            flagcxDataType,
            flagcxReduceOp,
            root,
            handler->comm,
            stream);

        // Create a future to track the reduce operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::REDUCE, std::move(future));

    }

    c10::intrusive_ptr<Work> BackendFlagcx::reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ReduceScatterOptions& opts)
    {
        auto outputTensor = outputTensors.back();
        auto inputTensors_ = inputTensors.back();
        auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
        auto flagcxReduceOp = getFlagcxReduceOp(opts.reduceOp, outputTensor, flagcxDataType, handler->comm);
        check_device(outputTensor.device(), inputTensors_[0].device());
        initComm(outputTensor.device());

        if (!check_same_size(inputTensors_))
        {
            throw std::runtime_error("flagcx only support same size reducescatter operation");
        }
        else
        {
            // Flatten a vector of tensors into a single, stacked tensor.
            at::Tensor inputFlattened = newLikeFlat(inputTensors_);

            // Sync the underlying stream with flagcx stream.
            syncStream(outputTensor.device());

            // Perform the reducescatter operation
            flagcxReduceScatter(
                inputFlattened.data_ptr(),
                outputTensor.data_ptr(),
                outputTensor.numel(),
                flagcxDataType,
                flagcxReduceOp,
                handler->comm,
                stream);

            // Unflatten the flattened tensor back into a vector of tensors.
            for (const auto j : c10::irange(inputTensors_.size()))
            {
                inputFlattened[j].copy_(inputTensors_[j], true);
            }
        }

        // Create a future to track the reducescatter operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensor));
        return c10::make_intrusive<WorkFlagcx>(OpType::REDUCE_SCATTER, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ScatterOptions& opts)
    {
        auto& outputTensor = outputTensors.back();
        auto& inputTensors_ = inputTensors.back();
        auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
        initComm(outputTensor.device());
        syncStream(outputTensor.device());

        auto root = opts.rootRank;

        // Flatten a vector of tensors into a single, stacked tensor.
        at::Tensor inputFlattened = newLikeFlat(inputTensors_);

        // Perform the scatter operation
        flagcxScatter(
            inputFlattened.data_ptr(),
            outputTensor.data_ptr(),
            outputTensor.numel(),
            flagcxDataType,
            root,
            handler->comm,
            stream);

        // Unflatten the flattened tensor back into a vector of tensors.
        for (const auto j : c10::irange(inputTensors_.size()))
        {
            inputFlattened[j].copy_(inputTensors_[j], true);
        }

        // Create a future to track the scatter operation
        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensor));
        return c10::make_intrusive<WorkFlagcx>(OpType::SCATTER, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::send(
        std::vector<at::Tensor>& tensors,
        int dstRank,
        int tag)
    {
        auto &tensor = tensors.back();
        auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
        initComm(tensor.device());
        syncStream(tensor.device());

        // Perform the send operation
        flagcxSend(
            tensor.data_ptr(),
            tensor.numel(),
            flagcxDataType,
            dstRank,
            handler->comm,
            stream);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::SEND, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::recv(
        std::vector<at::Tensor>& tensors,
        int srcRank,
        int tag)
    {
        auto &tensor = tensors.back();
        auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
        initComm(tensor.device());
        syncStream(tensor.device());

        // Perform the recv operation
        flagcxRecv(
            tensor.data_ptr(),
            tensor.numel(),
            flagcxDataType,
            srcRank,
            handler->comm,
            stream);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkFlagcx>(OpType::RECV, std::move(future));
    }

    c10::intrusive_ptr<Work> BackendFlagcx::recvAnysource(
        std::vector<at::Tensor>& tensors,
        int tag)
    {
        throw std::runtime_error("BackendFlagcx does not support recvAnysource");
    }

    c10::intrusive_ptr<Backend> BackendFlagcx::createBackendFlagcx(
        const c10::intrusive_ptr<::c10d::Store>& store,
        int rank,
        int size,
        const std::chrono::duration<float>& /* unused */)
    {
        return c10::make_intrusive<BackendFlagcx>(store, rank, size);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("createBackendFlagcx", &BackendFlagcx::createBackendFlagcx);
    }

} // namespace c10d