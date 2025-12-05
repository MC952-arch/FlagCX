#include "collectives.h"
#include "group.h"
#include "net.h"
#include "transport.h"
#include "type.h"

inline int flagcxFindP2pRound(flagcxHeteroComm_t comm, int isSendNotRecv,
                              int peer) {
  int round = 0;
  while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
                                : comm->p2pSchedule[round].recvRank)) {
    round += 1;
  }
  return round;
}

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm,
                                flagcxStream_t stream) {
  flagcxHeteroGroupStart();
  int round = flagcxFindP2pRound(comm, 1, peer);
  int channelId = round % MAXCHANNELS;
  if (comm->channels[channelId].peers[peer]->send[0].connected == 0) {
    comm->connectSend[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)sendbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = channelId;
  p2p->dtype = datatype;
  p2p->stream = stream;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm,
                                flagcxStream_t stream) {
  flagcxHeteroGroupStart();
  int round = flagcxFindP2pRound(comm, 0, peer);
  int channelId = round % MAXCHANNELS;
  if (comm->channels[channelId].peers[peer]->recv[0].connected == 0) {
    comm->connectRecv[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)recvbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = channelId;
  p2p->dtype = datatype;
  p2p->stream = stream;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}