#!/usr/bin/env python3
"""
Performance comparison script between old sequential QR detection and new parallel version.
This script benchmarks both implementations side by side.
"""

import time
import statistics
import sys
import os
from typing import List, Tuple
import importlib.util

def load_module_from_file(filepath: str, module_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def benchmark_detector(detector_class, detector_name: str, test_images: List[str], num_runs: int = 3):
    """
    Benchmark a QR code detector implementation.
    
    Args:
        detector_class: The QRCodeDetector class to test
        detector_name: Name for display purposes
        test_images: List of image paths to test
        num_runs: Number of runs per image for averaging
    
    Returns:
        dict: Results including timing and QR codes found
    """
    detector = detector_class()
    results = {
        'name': detector_name,
        'total_time': 0,
        'total_qr_codes': 0,
        'image_results': [],
        'avg_time_per_image': 0,
        'times': []
    }
    
    print(f"\n🔍 Testing {detector_name}")
    print("=" * 50)
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Skipping {image_path} (not found)")
            continue
            
        image_name = os.path.basename(image_path)
        print(f"📸 Processing: {image_name}")
        
        run_times = []
        qr_codes_found = 0
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Load image
            image = detector.load_image(image_path)
            if image is None:
                print(f"   ❌ Failed to load image")
                break
                
            # Detect QR codes
            qr_codes = detector.detect_qr_codes(image)
            
            end_time = time.time()
            run_time = end_time - start_time
            run_times.append(run_time)
            
            if run == 0:  # Only count QR codes from first run
                qr_codes_found = len(qr_codes)
        
        if run_times:
            avg_time = statistics.mean(run_times)
            min_time = min(run_times)
            max_time = max(run_times)
            
            results['total_time'] += sum(run_times)
            results['total_qr_codes'] += qr_codes_found
            results['times'].extend(run_times)
            results['image_results'].append({
                'image': image_name,
                'qr_codes': qr_codes_found,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time
            })
            
            print(f"   ✅ Found {qr_codes_found} QR code(s)")
            print(f"   ⏱️  Time: {avg_time:.3f}s (±{(max_time-min_time)/2:.3f}s)")
    
    if results['times']:
        results['avg_time_per_image'] = statistics.mean(results['times'])
    
    return results

def compare_implementations(old_file: str, new_file: str, test_images: List[str], num_runs: int = 3):
    """
    Compare old and new implementations side by side.
    """
    print("🚀 QR Code Detector Performance Comparison")
    print("=" * 60)
    print(f"📊 Testing {len(test_images)} images with {num_runs} runs each")
    
    # Load both implementations
    try:
        old_module = load_module_from_file(old_file, "old_detector")
        new_module = load_module_from_file(new_file, "new_detector")
    except Exception as e:
        print(f"❌ Error loading modules: {e}")
        return
    
    # Run benchmarks
    old_results = benchmark_detector(old_module.QRCodeDetector, "Old Sequential Version", test_images, num_runs)
    new_results = benchmark_detector(new_module.QRCodeDetector, "New Parallel Version", test_images, num_runs)
    
    # Compare results
    print("\n🏁 COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"📊 QR Codes Detected:")
    print(f"   Old Version: {old_results['total_qr_codes']}")
    print(f"   New Version: {new_results['total_qr_codes']}")
    if old_results['total_qr_codes'] == new_results['total_qr_codes']:
        print("   ✅ Both found the same number of QR codes")
    else:
        diff = new_results['total_qr_codes'] - old_results['total_qr_codes']
        print(f"   {'✅' if diff >= 0 else '⚠️'} New version found {abs(diff)} {'more' if diff > 0 else 'fewer'} QR codes")
    
    print(f"\n⏱️ Performance Comparison:")
    print(f"   Old Version Total Time: {old_results['total_time']:.3f}s")
    print(f"   New Version Total Time: {new_results['total_time']:.3f}s")
    
    if old_results['total_time'] > 0:
        speedup = old_results['total_time'] / new_results['total_time']
        time_saved = old_results['total_time'] - new_results['total_time']
        percent_faster = ((old_results['total_time'] - new_results['total_time']) / old_results['total_time']) * 100
        
        if speedup > 1:
            print(f"   🚀 New version is {speedup:.2f}x faster")
            print(f"   💨 Saved {time_saved:.3f}s ({percent_faster:.1f}% faster)")
        elif speedup < 1:
            print(f"   ⚠️ New version is {1/speedup:.2f}x slower")
            print(f"   🐌 Takes {abs(time_saved):.3f}s longer ({abs(percent_faster):.1f}% slower)")
        else:
            print(f"   ⚖️ Both versions perform similarly")
    
    print(f"\n📈 Average Time Per Image:")
    print(f"   Old Version: {old_results['avg_time_per_image']:.3f}s")
    print(f"   New Version: {new_results['avg_time_per_image']:.3f}s")
    
    # Detailed breakdown
    print(f"\n📋 Detailed Image Breakdown:")
    print(f"{'Image':<20} {'Old (s)':<10} {'New (s)':<10} {'QR Codes':<10} {'Speedup':<10}")
    print("-" * 70)
    
    for i, old_img in enumerate(old_results['image_results']):
        if i < len(new_results['image_results']):
            new_img = new_results['image_results'][i]
            speedup = old_img['avg_time'] / new_img['avg_time'] if new_img['avg_time'] > 0 else 0
            print(f"{old_img['image'][:19]:<20} {old_img['avg_time']:<10.3f} {new_img['avg_time']:<10.3f} {new_img['qr_codes']:<10} {speedup:<10.2f}x")
    
    # Analysis insights
    print(f"\n💡 Analysis:")
    if new_results['total_time'] < old_results['total_time']:
        print("   ✅ Parallel processing successfully improved performance")
        print("   ⚡ Multiple preprocessing techniques now run simultaneously")
        print("   🎯 Better CPU core utilization")
    
    if new_results['total_qr_codes'] >= old_results['total_qr_codes']:
        print("   ✅ No QR codes lost in optimization")
        print("   🔍 Maintains detection quality while improving speed")
    
    if any(new['qr_codes'] > old['qr_codes'] for new, old in zip(new_results['image_results'], old_results['image_results'])):
        print("   🎉 Some images show improved detection (found more QR codes)")

if __name__ == "__main__":
    # Test images
    test_images = [
        "Cornucopia 2025 Photos/IMG_2200.HEIC",
        "Cornucopia 2025 Photos/IMG_2202.HEIC",
        "Cornucopia 2025 Photos/IMG_2201.HEIC",
        "Cornucopia 2025 Photos/IMG_2203.HEIC",
    ]
    
    # Filter to only existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if len(existing_images) < 2:
        print("❌ Need at least 2 test images for meaningful comparison")
        print("📁 Available images:")
        photo_dir = "Cornucopia 2025 Photos"
        if os.path.exists(photo_dir):
            for f in sorted(os.listdir(photo_dir))[:10]:
                if f.lower().endswith(('.heic', '.jpg', '.jpeg', '.png')):
                    print(f"   📸 {f}")
    else:
        compare_implementations(
            old_file="old.py",
            new_file="qr_code_detector.py", 
            test_images=existing_images,
            num_runs=2  # 2 runs for faster testing
        )
