#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "yaml"

config_dir = ARGV.fetch(0, ".github/configs")
config_files = Dir.glob(File.join(config_dir, "*.yml")).sort
abort "No platform configs found in #{config_dir}" if config_files.empty?

required_keys = %w[platform display_name runner_labels container set_env unit_test_suites]
platforms = config_files.map do |path|
  contents = File.read(path)
  config = begin
    YAML.safe_load(
      contents,
      permitted_classes: [],
      permitted_symbols: [],
      aliases: false,
      filename: path
    )
  rescue ArgumentError
    # Compatibility with the older Ruby/Psych available on some self-hosted
    # runners, whose safe_load API only accepts positional arguments.
    YAML.safe_load(contents, [], [], false, path)
  end
  missing = required_keys.reject { |key| config.key?(key) }
  abort "#{path}: missing required keys: #{missing.join(', ')}" unless missing.empty?

  container = config.fetch("container")
  %w[image options].each do |key|
    abort "#{path}: container.#{key} is required" unless container.key?(key)
  end

  set_env = config.fetch("set_env")
  abort "#{path}: set_env does not exist: #{set_env}" unless File.file?(set_env)

  runner_labels = config.fetch("runner_labels")
  suites = config.fetch("unit_test_suites")
  abort "#{path}: runner_labels must be a non-empty array" unless runner_labels.is_a?(Array) && !runner_labels.empty?
  abort "#{path}: unit_test_suites must be a non-empty array" unless suites.is_a?(Array) && !suites.empty?

  {
    "platform" => config.fetch("platform"),
    "display_name" => config.fetch("display_name"),
    "runs_on" => JSON.generate(runner_labels),
    "image" => container.fetch("image"),
    "container_options" => container.fetch("options"),
    "set_env" => set_env,
    "unit_test_suites" => JSON.generate(suites)
  }
end

duplicates = platforms.group_by { |platform| platform.fetch("platform") }
                      .select { |_name, entries| entries.length > 1 }
abort "Duplicate platform configs: #{duplicates.keys.join(', ')}" unless duplicates.empty?

matrix = JSON.generate({ "include" => platforms })
puts matrix

if ENV["GITHUB_OUTPUT"] && !ENV["GITHUB_OUTPUT"].empty?
  File.open(ENV.fetch("GITHUB_OUTPUT"), "a") do |output|
    output.puts "matrix=#{matrix}"
  end
end
