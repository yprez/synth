#!/usr/bin/env python3
"""
Comprehensive Synthesizer Performance Benchmark

This script benchmarks the performance of each effect in the QWERTY Synth
individually and in various combinations to identify performance bottlenecks
and optimize the audio processing pipeline.
"""

import numpy as np
import time

# Import synth modules
from qwerty_synth import config
from qwerty_synth import synth
from qwerty_synth.filter import apply_filter, reset_filter_state
from qwerty_synth.drive import apply_drive
from qwerty_synth.delay import Delay
from qwerty_synth.chorus import Chorus
from qwerty_synth.lfo import LFO


class PerformanceBenchmark:
    """Comprehensive performance benchmark for the synthesizer."""

    def __init__(self, duration=1.0, sample_rate=44100):
        """Initialize benchmark with test parameters."""
        self.duration = duration
        self.sample_rate = sample_rate
        self.frames = int(sample_rate * duration)
        self.results = {}

        # Configure synth for testing
        config.sample_rate = sample_rate
        config.volume = 0.5
        config.blocksize = 2048

        # Test signal setup
        self._setup_test_signals()

        # Initialize effects
        self.delay = Delay(sample_rate)
        self.chorus = Chorus(sample_rate)
        self.lfo = LFO()

    def _setup_test_signals(self):
        """Create various test signals for benchmarking."""
        t = np.linspace(0, self.duration, self.frames, dtype=np.float32)

        # Test signals
        self.signals = {
            'sine': np.sin(2 * np.pi * 440 * t),
            'complex': (np.sin(2 * np.pi * 440 * t) +
                       0.5 * np.sin(2 * np.pi * 880 * t) +
                       0.25 * np.sin(2 * np.pi * 1760 * t)),
            'noise': np.random.normal(0, 0.1, self.frames).astype(np.float32),
            'quiet': np.sin(2 * np.pi * 440 * t) * 0.01,  # Very quiet signal
            'loud': np.clip(np.sin(2 * np.pi * 440 * t) * 2.0, -1.0, 1.0)  # Clipped signal
        }

        # Normalize all signals
        for name, signal in self.signals.items():
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                self.signals[name] = signal / max_val * 0.5

    def _benchmark_test(self, test_name, test_func, iterations=10):
        """Run a benchmark test with timing and statistics."""
        print(f"Testing {test_name}...")
        times = []

        for i in range(iterations):
            start_time = time.perf_counter()
            test_func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        # Performance metrics
        samples_per_sec = self.frames / avg_time
        realtime_factor = samples_per_sec / self.sample_rate
        cpu_usage = (avg_time / self.duration) * 100

        result = {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'std_time_ms': std_time * 1000,
            'samples_per_sec': samples_per_sec,
            'realtime_factor': realtime_factor,
            'cpu_usage_percent': cpu_usage,
            'iterations': iterations
        }

        self.results[test_name] = result
        print(f"  Average: {avg_time*1000:.2f}ms ({realtime_factor:.1f}x RT, {cpu_usage:.1f}% CPU)")

    def benchmark_oscillator(self):
        """Benchmark oscillator generation performance."""
        print("\n=== OSCILLATOR BENCHMARK ===")

        waveforms = ['sine', 'square', 'triangle', 'sawtooth']
        frequencies = [220, 440, 880, 1760]  # A3, A4, A5, A6

        for waveform in waveforms:
            config.waveform_type = waveform

            for freq in frequencies:
                test_name = f"oscillator_{waveform}_{freq}Hz"

                def test_func():
                    osc = synth.Oscillator(freq, waveform)
                    output, filter_env = osc.generate(self.frames)

                self._benchmark_test(test_name, test_func)

    def benchmark_adsr(self):
        """Benchmark ADSR envelope performance."""
        print("\n=== ADSR ENVELOPE BENCHMARK ===")

        # Test different ADSR settings
        adsr_configs = [
            {'attack': 0.01, 'decay': 0.02, 'sustain': 0.7, 'release': 0.2},  # Fast
            {'attack': 0.1, 'decay': 0.3, 'sustain': 0.5, 'release': 1.0},   # Slow
            {'attack': 0.001, 'decay': 0.001, 'sustain': 1.0, 'release': 0.001},  # Instant
        ]

        for i, adsr_config in enumerate(adsr_configs):
            config.adsr.update(adsr_config)
            test_name = f"adsr_config_{i+1}"

            def test_func():
                osc = synth.Oscillator(440, 'sine')
                output, filter_env = osc.generate(self.frames)

            self._benchmark_test(test_name, test_func)

    def benchmark_filter(self):
        """Benchmark filter performance with different configurations."""
        print("\n=== FILTER BENCHMARK ===")

        filter_types = ['lowpass', 'highpass', 'bandpass', 'notch']
        topologies = ['svf', 'biquad']
        cutoffs = [500, 1000, 2000, 5000]
        resonances = [0.0, 0.5, 0.9]

        config.filter_enabled = True

        for topology in topologies:
            config.filter_topology = topology

            for filter_type in filter_types:
                config.filter_type = filter_type

                for cutoff in cutoffs:
                    config.filter_cutoff = cutoff

                    for resonance in resonances:
                        config.filter_resonance = resonance
                        test_name = f"filter_{topology}_{filter_type}_{cutoff}Hz_Q{resonance}"

                        def test_func():
                            reset_filter_state()
                            signal = self.signals['complex'].copy()
                            filtered = apply_filter(signal)

                        self._benchmark_test(test_name, test_func, iterations=5)

    def benchmark_drive(self):
        """Benchmark drive/distortion effect performance."""
        print("\n=== DRIVE BENCHMARK ===")

        drive_types = ['tanh', 'arctan', 'cubic', 'fuzz', 'asymmetric']
        gains = [1.0, 1.5, 2.0, 3.0]

        config.drive_on = True

        for drive_type in drive_types:
            config.drive_type = drive_type

            for gain in gains:
                config.drive_gain = gain
                test_name = f"drive_{drive_type}_gain{gain}"

                def test_func():
                    signal = self.signals['complex'].copy()
                    processed = apply_drive(signal)

                self._benchmark_test(test_name, test_func)

    def benchmark_delay(self):
        """Benchmark delay effect performance."""
        print("\n=== DELAY BENCHMARK ===")

        delay_times = [50, 100, 250, 500, 1000]  # ms
        feedback_levels = [0.0, 0.3, 0.6, 0.9]

        for delay_time in delay_times:
            for feedback in feedback_levels:
                test_name = f"delay_{delay_time}ms_fb{feedback}"

                def test_func():
                    delay = Delay(self.sample_rate, delay_time)
                    signal = self.signals['sine'].copy()
                    processed = delay.process_block(signal, feedback, 0.5)

                self._benchmark_test(test_name, test_func)

        # Test ping-pong delay
        def test_pingpong():
            delay = Delay(self.sample_rate, 250)
            signal_L = self.signals['sine'].copy()
            signal_R = self.signals['sine'].copy()
            out_L, out_R = delay.pingpong(signal_L, signal_R, 0.5, 0.3)

        self._benchmark_test("delay_pingpong", test_pingpong)

    def benchmark_chorus(self):
        """Benchmark chorus effect performance."""
        print("\n=== CHORUS BENCHMARK ===")

        rates = [0.5, 1.0, 2.0, 5.0]  # Hz
        depths = [0.002, 0.005, 0.010, 0.020]  # seconds
        voices = [1, 2, 3, 4]

        for voice_count in voices:
            for rate in rates:
                for depth in depths:
                    test_name = f"chorus_{voice_count}voices_{rate}Hz_{depth*1000:.0f}ms"

                    def test_func():
                        chorus = Chorus(self.sample_rate)
                        chorus.set_voices(voice_count)
                        chorus.set_rate(rate)
                        chorus.set_depth(depth)
                        chorus.set_mix(0.3)

                        signal_L = self.signals['complex'].copy()
                        signal_R = self.signals['complex'].copy()
                        out_L, out_R = chorus.process(signal_L, signal_R)

                    self._benchmark_test(test_name, test_func, iterations=3)

    def benchmark_lfo(self):
        """Benchmark LFO performance."""
        print("\n=== LFO BENCHMARK ===")

        targets = ['pitch', 'volume', 'cutoff']
        rates = [0.5, 2.0, 5.0, 10.0]  # Hz
        depths = [0.1, 0.3, 0.5, 1.0]

        config.lfo_enabled = True

        for target in targets:
            config.lfo_target = target

            for rate in rates:
                config.lfo_rate = rate

                for depth in depths:
                    config.lfo_depth = depth
                    test_name = f"lfo_{target}_{rate}Hz_depth{depth}"

                    def test_func():
                        lfo = LFO()
                        lfo_signal = lfo.generate(self.frames)

                        if target == 'pitch':
                            freq_array = np.full(self.frames, 440.0)
                            modulated = lfo.apply_pitch_modulation(freq_array, lfo_signal)
                        elif target == 'volume':
                            env = np.full(self.frames, 0.5)
                            modulated = lfo.apply_amplitude_modulation(env, lfo_signal)
                        elif target == 'cutoff':
                            cutoff_mod = lfo.get_cutoff_modulation(self.frames)

                    self._benchmark_test(test_name, test_func)

    def benchmark_combinations(self):
        """Benchmark combinations of effects."""
        print("\n=== EFFECT COMBINATIONS BENCHMARK ===")

        # Test realistic effect chains
        effect_chains = [
            {"name": "basic", "effects": ["oscillator", "filter"]},
            {"name": "distorted", "effects": ["oscillator", "drive", "filter"]},
            {"name": "ambient", "effects": ["oscillator", "filter", "chorus", "delay"]},
            {"name": "full_chain", "effects": ["oscillator", "lfo", "filter", "drive", "chorus", "delay"]},
        ]

        for chain in effect_chains:
            test_name = f"chain_{chain['name']}"

            # Setup for chain
            config.filter_enabled = "filter" in chain["effects"]
            config.drive_on = "drive" in chain["effects"]
            config.chorus_enabled = "chorus" in chain["effects"]
            config.delay_enabled = "delay" in chain["effects"]
            config.lfo_enabled = "lfo" in chain["effects"]

            # Configure reasonable settings
            config.filter_cutoff = 2000
            config.filter_resonance = 0.3
            config.drive_gain = 1.5
            config.chorus_mix = 0.2
            config.delay_mix = 0.1
            config.delay_time_ms = 250
            config.lfo_rate = 2.0
            config.lfo_depth = 0.2

            def test_func():
                # Simulate real-time audio processing
                buffer = np.zeros(config.blocksize)

                # Create active note
                osc = synth.Oscillator(440, 'sine')

                # Process multiple blocks to simulate real performance
                total_frames = 0
                while total_frames < self.frames:
                    frames_to_process = min(config.blocksize, self.frames - total_frames)

                    # Generate oscillator output
                    wave, filter_env = osc.generate(frames_to_process)
                    buffer[:frames_to_process] = wave

                    # Apply effects in order
                    if config.filter_enabled:
                        buffer[:frames_to_process] = apply_filter(buffer[:frames_to_process])

                    if config.drive_on:
                        buffer[:frames_to_process] = apply_drive(buffer[:frames_to_process])

                    if config.chorus_enabled:
                        signal_L = buffer[:frames_to_process].copy()
                        signal_R = buffer[:frames_to_process].copy()
                        chorus = Chorus(self.sample_rate)
                        out_L, out_R = chorus.process(signal_L, signal_R)
                        buffer[:frames_to_process] = (out_L + out_R) / 2

                    if config.delay_enabled:
                        delay = Delay(self.sample_rate, config.delay_time_ms)
                        buffer[:frames_to_process] = delay.process_block(
                            buffer[:frames_to_process], 0.3, 0.2)

                    total_frames += frames_to_process

            self._benchmark_test(test_name, test_func, iterations=3)

    def benchmark_audio_callback(self):
        """Benchmark the main audio callback function."""
        print("\n=== AUDIO CALLBACK BENCHMARK ===")

        # Test with different numbers of active notes
        note_counts = [1, 2, 4, 8, 16]

        for note_count in note_counts:
            test_name = f"audio_callback_{note_count}_notes"

            # Setup active notes
            config.active_notes.clear()
            for i in range(note_count):
                freq = 220 * (2 ** (i / 12))  # Chromatic scale
                key = f"test_key_{i}"
                config.active_notes[key] = synth.Oscillator(freq, 'sine')

            # Enable various effects for realistic testing
            config.filter_enabled = True
            config.drive_on = True
            config.chorus_enabled = True
            config.delay_enabled = True

            def test_func():
                # Simulate audio callback
                outdata = np.zeros((config.blocksize, 2))
                synth.audio_callback(outdata, config.blocksize, None, None)

            self._benchmark_test(test_name, test_func, iterations=5)

    def print_summary(self):
        """Print a comprehensive performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)

        # Group results by category
        categories = {
            'Oscillator': [k for k in self.results.keys() if k.startswith('oscillator_')],
            'ADSR': [k for k in self.results.keys() if k.startswith('adsr_')],
            'Filter': [k for k in self.results.keys() if k.startswith('filter_')],
            'Drive': [k for k in self.results.keys() if k.startswith('drive_')],
            'Delay': [k for k in self.results.keys() if k.startswith('delay_')],
            'Chorus': [k for k in self.results.keys() if k.startswith('chorus_')],
            'LFO': [k for k in self.results.keys() if k.startswith('lfo_')],
            'Combinations': [k for k in self.results.keys() if k.startswith('chain_')],
            'Audio Callback': [k for k in self.results.keys() if k.startswith('audio_callback_')]
        }

        for category, tests in categories.items():
            if not tests:
                continue

            print(f"\n{category}:")
            print("-" * 40)

            # Calculate category statistics
            times = [self.results[test]['avg_time_ms'] for test in tests]
            rt_factors = [self.results[test]['realtime_factor'] for test in tests]
            cpu_usages = [self.results[test]['cpu_usage_percent'] for test in tests]

            if times:
                print(f"  Average Time: {np.mean(times):.2f}ms (range: {np.min(times):.2f}-{np.max(times):.2f}ms)")
                print(f"  Average RT Factor: {np.mean(rt_factors):.1f}x (range: {np.min(rt_factors):.1f}-{np.max(rt_factors):.1f}x)")
                print(f"  Average CPU Usage: {np.mean(cpu_usages):.1f}% (range: {np.min(cpu_usages):.1f}-{np.max(cpu_usages):.1f}%)")

                # Find best and worst performers
                best_idx = np.argmax(rt_factors)
                worst_idx = np.argmin(rt_factors)
                print(f"  Best performer: {tests[best_idx]} ({rt_factors[best_idx]:.1f}x RT)")
                print(f"  Worst performer: {tests[worst_idx]} ({rt_factors[worst_idx]:.1f}x RT)")

        # Overall performance analysis
        print("\nOVERALL ANALYSIS:")
        print("-" * 40)
        all_rt_factors = [result['realtime_factor'] for result in self.results.values()]
        all_cpu_usages = [result['cpu_usage_percent'] for result in self.results.values()]

        print(f"Total tests performed: {len(self.results)}")
        print(f"Overall average RT factor: {np.mean(all_rt_factors):.1f}x")
        print(f"Overall average CPU usage: {np.mean(all_cpu_usages):.1f}%")
        print(f"Tests below 1x RT factor: {sum(1 for rt in all_rt_factors if rt < 1.0)}")

        # Performance recommendations
        problematic_tests = [name for name, result in self.results.items()
                           if result['realtime_factor'] < 1.0]
        if problematic_tests:
            print("\nPROBLEMATIC TESTS (< 1x RT):")
            for test in problematic_tests:
                result = self.results[test]
                print(f"  {test}: {result['realtime_factor']:.2f}x RT ({result['cpu_usage_percent']:.1f}% CPU)")

    def save_results(self, filename="synth_benchmark_results.txt"):
        """Save detailed results to a file."""
        with open(filename, 'w') as f:
            f.write("QWERTY Synth Performance Benchmark Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Test Duration: {self.duration}s\n")
            f.write(f"Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"Total Frames: {self.frames}\n\n")

            for test_name, result in self.results.items():
                f.write(f"{test_name}:\n")
                f.write(f"  Average Time: {result['avg_time_ms']:.3f}ms\n")
                f.write(f"  Min/Max Time: {result['min_time_ms']:.3f}/{result['max_time_ms']:.3f}ms\n")
                f.write(f"  Std Dev: {result['std_time_ms']:.3f}ms\n")
                f.write(f"  RT Factor: {result['realtime_factor']:.2f}x\n")
                f.write(f"  CPU Usage: {result['cpu_usage_percent']:.2f}%\n")
                f.write(f"  Iterations: {result['iterations']}\n\n")

        print(f"\nDetailed results saved to {filename}")

    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("Starting Comprehensive Synthesizer Performance Benchmark")
        print(f"Duration: {self.duration}s, Sample Rate: {self.sample_rate} Hz")
        print("="*80)

        # Run all benchmarks
        self.benchmark_oscillator()
        self.benchmark_adsr()
        self.benchmark_filter()
        self.benchmark_drive()
        self.benchmark_delay()
        self.benchmark_chorus()
        self.benchmark_lfo()
        self.benchmark_combinations()
        self.benchmark_audio_callback()

        # Print results
        self.print_summary()
        self.save_results()


def main():
    """Main function to run the benchmark."""
    try:
        # Create and run benchmark
        benchmark = PerformanceBenchmark(duration=0.5, sample_rate=44100)  # Shorter duration for testing
        benchmark.run_full_benchmark()

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
