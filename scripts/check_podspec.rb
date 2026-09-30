#!/usr/bin/env ruby
# scripts/check_podspec.rb — validate Sofuu.podspec without CocoaPods.
#
# A malformed podspec is invisible to every other gate in this repo (it is
# not Rust, not C, not JS) and breaks the install for every consumer the
# moment they run `pod install`. This is a minimal Pod::Spec shim that
# evaluates the spec and asserts the fields CocoaPods requires are present
# and sane, so a typo fails CI instead of a customer's machine.
#
# It is NOT a replacement for `pod lib lint` (which does a real build); it is
# the cheap gate that catches the common failures. Run `pod lib lint` before
# an actual release.
#
# Usage: ruby scripts/check_podspec.rb
# Minimal Pod::Spec shim so `ruby` alone can validate the podspec without
# CocoaPods installed. Mirrors the DSL the spec uses; raises on anything it
# does not model, so a typo in the podspec still fails.
module Pod
  class Spec
    def self.new(&blk)
      s = allocate
      s.send(:initialize)
      blk.call(s)
      s.validate!
      s
    end

    def initialize
      @attrs = {}
      @ios = Platform.new("ios")
      @osx = Platform.new("osx")
      @source_files = []
      @vendored_frameworks = nil
    end
    attr_reader :ios, :osx, :source_files, :vendored_frameworks

    class Platform
      def initialize(n); @n = n; end
      def deployment_target(v = :__get__)
        return @target if v == :__get__
        @target = v
      end
      def deployment_target=(v); @target = v; end
    end

    %w[name version summary description homepage license static_framework
       requires_arc pod_target_xcconfig].each do |m|
      define_method(m) { |v = :__get__| v == :__get__ ? @attrs[m] : @attrs[m] = v }
      define_method("#{m}=") { |v| @attrs[m] = v }
    end
    def author(h = :__get__); h == :__get__ ? @attrs["author"] : @attrs["author"] = h; end
    def author=(v); @attrs["author"] = v; end

    def source(h = :__get__); h == :__get__ ? @attrs["source"] : @attrs["source"] = h; end
    def source=(v); @attrs["source"] = v; end
    def vendored_frameworks(*f); f.empty? ? @vendored_frameworks : (@vendored_frameworks = f.flatten.join(", ")); end
    def vendored_frameworks=(v); @vendored_frameworks = v; end
    def source_files(*f); f.flatten.each { |x| @source_files << x }; end
    def source_files=(v); @source_files = Array(v); end
    def consumerProguardFiles(*f); @attrs["proguard"] = f.flatten.join(", "); end
    def test_spec(name); yield SpecSpec.new; end
    class SpecSpec
      def source_files(*f); f.flatten.each { |x| @f = x }; end
      def source_files=(v); @f = v; end
    end

    def validate!
      required = %w[name version summary homepage license author source]
      missing = required.select { |r| @attrs[r].nil? || @attrs[r].to_s.empty? }
      raise "podspec missing: #{missing.join(', ')}" unless missing.empty?
      raise "podspec needs a source URL" unless @attrs["source"].is_a?(Hash) && @attrs["source"][:http]
      raise "podspec must vendor the xcframework" unless @vendored_frameworks.to_s.include?("xcframework")
      puts "✓ podspec valid: #{@attrs['name']} #{@attrs['version']} (ios #{ios.deployment_target}, osx #{osx.deployment_target})"
    end
  end
end

spec = eval(File.read("Sofuu.podspec"), binding, "Sofuu.podspec")
