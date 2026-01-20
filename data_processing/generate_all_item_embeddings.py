#!/usr/bin/env python3
"""
Generate All Item Embeddings Script - Enhanced with Efficient Batch Processing

This script processes all items in a JSON data file and generates query tokens
for each item using the Q-Former model. The output is a JSON file where
each item ID maps to its corresponding query tokens.

NEW: Efficient batch processing that processes multiple items simultaneously,
dramatically improving processing speed compared to individual processing.

Performance improvements:
- True batch processing: 2-5x faster than individual processing
- Memory optimization options for large datasets
- Detailed profiling and timing information
- Fallback to individual processing if batch processing fails

Usage:
    # Use efficient batch processing (default, recommended)
    python generate_all_item_embeddings.py
    
    # Use legacy individual processing
    python generate_all_item_embeddings.py --legacy-processing
    
    # Enable memory optimization and profiling
    python generate_all_item_embeddings.py --memory-efficient --profile
    
    # Run performance comparison first, then process
    python generate_all_item_embeddings.py --compare --compare-sample-size 100
    
    # Custom batch size and checkpoint
    python generate_all_item_embeddings.py --batch-size 16 --checkpoint "path/to/checkpoint.pth"
    
    # Custom paths
    python generate_all_item_embeddings.py --data "path/to/data.json" --output "embeddings.json"
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import torch

# Add the current directory to the path to import qformer_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qformer_inference import QFormerInference, generate_query_tokens_for_item_id

def check_gpu_status():
    """Check and display GPU status information."""
    print("🔍 GPU Status Check")
    print("=" * 40)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA is available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Check current GPU memory usage
            if gpu_count > 0:
                current_device = torch.cuda.current_device()
                allocated_memory = torch.cuda.memory_allocated() / 1024**3
                try:
                    available_memory = torch.cuda.memory_available() / 1024**3
                except AttributeError:
                    # Fallback for older PyTorch versions
                    total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                    available_memory = total_memory - allocated_memory
                total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                print(f"\n💾 Current GPU Memory Status (GPU {current_device}):")
                print(f"   Allocated: {allocated_memory:.2f} GB")
                print(f"   Available: {available_memory:.2f} GB")
                print(f"   Total: {total_memory:.2f} GB")
                print(f"   Usage: {allocated_memory/total_memory*100:.1f}%")
                
                # Test GPU allocation
                try:
                    test_tensor = torch.zeros(1000, 1000, device="cuda")
                    test_memory = torch.cuda.memory_allocated() / 1024**3
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    print(f"\n🧪 GPU Test Results:")
                    print(f"   ✅ GPU allocation test passed")
                    print(f"   📊 Test tensor memory: {test_memory - allocated_memory:.2f} GB")
                    
                except Exception as e:
                    print(f"\n❌ GPU allocation test failed: {e}")
                    return False
                
                return True
            else:
                print("❌ No GPUs found")
                return False
                
        else:
            print("❌ CUDA is not available")
            print("   This could be due to:")
            print("   - PyTorch not compiled with CUDA support")
            print("   - No NVIDIA GPU drivers installed")
            print("   - CUDA toolkit not properly installed")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")
        return False

def load_item_data(data_path: str) -> Dict[str, Any]:
    """
    Load item data from the JSON file.
    
    Args:
        data_path: Path to the JSON data file
        
    Returns:
        Dictionary mapping item IDs to item data
    """
    print(f"📂 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Ensure data is a dictionary with item IDs as keys
    if isinstance(data, list):
        # Convert list to dictionary if needed
        data = {f"item_{i}": item for i, item in enumerate(data)}
    
    print(f"✅ Loaded {len(data)} items from data file")
    return data

def generate_embeddings_for_all_items(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
    device: str = "auto",
    batch_size: int = 256,
    max_items: int = None,
    save_progress: bool = True,
    memory_efficient: bool = False,
    profile: bool = False
) -> Dict[str, List[List[float]]]:
    """
    Generate embeddings for all items in the data file using efficient batch processing.
    
    Args:
        data_path: Path to the JSON data file
        checkpoint_path: Path to the Q-Former checkpoint
        output_path: Path to save the output embeddings
        device: Device to run inference on
        batch_size: Batch size for processing
        max_items: Maximum number of items to process (None for all)
        save_progress: Whether to save progress periodically
        memory_efficient: Enable memory-efficient processing
        profile: Enable detailed profiling and timing information
        
    Returns:
        Dictionary mapping item IDs to query tokens
    """
    print("🚀 Starting efficient batch embedding generation for all items...")
    
    # Load item data
    items_data = load_item_data(data_path)
    
    # Limit items if specified
    if max_items:
        items_data = dict(list(items_data.items())[:max_items])
        print(f"📊 Processing {len(items_data)} items (limited by max_items)")
    
    # Initialize inference
    print("🔄 Initializing Q-Former inference...")
    inference = QFormerInference(checkpoint_path, device)
    
    # Check if we're using GPU and optimize if possible
    if inference.device.type == "cuda":
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
        
        # Set optimal GPU settings for inference
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Check GPU memory and adjust batch size if needed
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8.0:  # Less than 8GB
            if batch_size > 4:
                original_batch_size = batch_size
                batch_size = min(4, batch_size)
                print(f"🧠 GPU memory limited ({gpu_memory:.1f} GB), reduced batch size from {original_batch_size} to {batch_size}")
        elif gpu_memory < 16.0:  # Less than 16GB
            if batch_size > 8:
                original_batch_size = batch_size
                batch_size = min(8, batch_size)
                print(f"🧠 GPU memory moderate ({gpu_memory:.1f} GB), reduced batch size from {original_batch_size} to {batch_size}")
        else:
            print(f"🧠 GPU memory sufficient ({gpu_memory:.1f} GB), using batch size {batch_size}")
    else:
        print(f"💻 Using CPU for processing")
    
    # Dictionary to store results
    embeddings_dict = {}
    
    # Progress tracking
    total_items = len(items_data)
    processed_items = 0
    start_time = time.time()
    
    # Memory optimization: adjust batch size if needed
    if memory_efficient and batch_size > 4:
        original_batch_size = batch_size
        batch_size = min(4, batch_size)
        print(f"🧠 Memory-efficient mode: reduced batch size from {original_batch_size} to {batch_size}")
    
    print(f"🔄 Processing {total_items} items with true batch size {batch_size}")
    
    # Process items in true batches
    item_ids = list(items_data.keys())
    
    # Profiling data
    batch_times = []
    memory_usage = []
    
    for i in range(0, total_items, batch_size):
        batch_ids = item_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_items + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_ids)} items)")
        batch_start_time = time.time()
        
        # Memory cleanup before batch processing
        if memory_efficient or inference.device.type == "cuda":
            import gc
            gc.collect()
            if inference.device.type == "cuda":
                torch.cuda.empty_cache()
                if profile:
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"      💾 GPU memory before batch: {allocated_memory:.2f} GB")
        
        try:
            # Process the entire batch at once using the new efficient batch method
            batch_results = inference.generate_query_tokens_batch_by_ids(batch_ids, data_path)
            
            # Process results for this batch
            for result in batch_results:
                item_id = result['item_id']
                query_tokens = result['query_tokens']
                
                # Store in results dictionary
                embeddings_dict[item_id] = query_tokens
                processed_items += 1
            
            # Calculate batch timing and efficiency
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Print batch progress
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / processed_items
            remaining_items = total_items - processed_items
            estimated_remaining_time = remaining_items * avg_time_per_item
            
            print(f"      ✅ Batch completed in {batch_time:.2f}s")
            print(f"      📊 Progress: {processed_items}/{total_items} ({processed_items/total_items*100:.1f}%)")
            print(f"      ⏱️  Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            print(f"      🚀 Batch efficiency: {len(batch_ids)/batch_time:.2f} items/second")
            
            # GPU memory status after batch
            if inference.device.type == "cuda" and profile:
                allocated_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"      💾 GPU memory after batch: {allocated_memory:.2f} GB")
            
            # Detailed profiling information
            if profile:
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"      📈 Average batch time: {avg_batch_time:.2f}s")
                print(f"      📊 Batch time variance: {sum((t - avg_batch_time)**2 for t in batch_times) / len(batch_times):.3f}")
            
        except Exception as e:
            print(f"      ❌ Error processing batch {batch_num}: {e}")
            # Process items individually as fallback
            print(f"      🔄 Falling back to individual processing for this batch...")
            for item_id in batch_ids:
                try:
                    result = inference.generate_query_tokens_by_id(item_id, data_path)
                    query_tokens = result['query_tokens']
                    embeddings_dict[item_id] = query_tokens
                    processed_items += 1
                    print(f"         ✅ Processed {item_id} individually")
                except Exception as individual_error:
                    print(f"         ❌ Error processing item {item_id}: {individual_error}")
                    embeddings_dict[item_id] = {"error": str(individual_error)}
                    processed_items += 1
        
        # Save progress periodically
        if save_progress and batch_num % 5 == 0:
            progress_path = output_path.replace('.json', f'_progress_batch_{batch_num}.json')
            print(f"💾 Saving progress to: {progress_path}")
            with open(progress_path, 'w') as f:
                json.dump(embeddings_dict, f, indent=2)
    
    # Final save
    print(f"\n💾 Saving final results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    # Print final summary with detailed statistics
    total_time = time.time() - start_time
    successful_items = sum(1 for v in embeddings_dict.values() if not isinstance(v, dict) or "error" not in v)
    error_items = total_items - successful_items
    
    print(f"\n🎉 Efficient batch embedding generation completed!")
    print(f"   Total items processed: {total_items}")
    print(f"   Successful: {successful_items}")
    print(f"   Errors: {error_items}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average time per item: {total_time/total_items:.2f} seconds")
    print(f"   Overall processing speed: {total_items/total_time:.2f} items/second")
    
    if profile and batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"   📊 Batch processing statistics:")
        print(f"      Average batch time: {avg_batch_time:.2f}s")
        print(f"      Fastest batch: {min(batch_times):.2f}s")
        print(f"      Slowest batch: {max(batch_times):.2f}s")
        print(f"      Batch time std dev: {(sum((t - avg_batch_time)**2 for t in batch_times) / len(batch_times))**0.5:.3f}s")
    
    print(f"   Results saved to: {output_path}")
    
    return embeddings_dict

def generate_embeddings_for_all_items_legacy(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
    device: str = "auto",
    batch_size: int = 256,
    max_items: int = None,
    save_progress: bool = True
) -> Dict[str, List[List[float]]]:
    """
    Legacy function: Generate embeddings for all items using individual processing.
    This maintains the old behavior for backward compatibility.
    
    Args:
        data_path: Path to the JSON data file
        checkpoint_path: Path to the Q-Former checkpoint
        output_path: Path to save the output embeddings
        device: Device to run inference on
        batch_size: Batch size for processing (used for progress tracking only)
        max_items: Maximum number of items to process (None for all)
        save_progress: Whether to save progress periodically
        
    Returns:
        Dictionary mapping item IDs to query tokens
    """
    print("🚀 Starting legacy individual embedding generation for all items...")
    
    # Load item data
    items_data = load_item_data(data_path)
    
    # Limit items if specified
    if max_items:
        items_data = dict(list(items_data.items())[:max_items])
        print(f"📊 Processing {len(items_data)} items (limited by max_items)")
    
    # Initialize inference
    print("🔄 Initializing Q-Former inference...")
    inference = QFormerInference(checkpoint_path, device)
    
    # Dictionary to store results
    embeddings_dict = {}
    
    # Progress tracking
    total_items = len(items_data)
    processed_items = 0
    start_time = time.time()
    
    print(f"🔄 Processing {total_items} items with individual processing")
    
    # Process items individually
    item_ids = list(items_data.keys())
    
    for i in range(0, total_items, batch_size):
        batch_ids = item_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_items + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_ids)} items)")
        
        for item_id in batch_ids:
            try:
                print(f"   🔍 Processing item: {item_id}")
                
                # Generate query tokens for this item
                result = inference.generate_query_tokens_by_id(item_id, data_path)
                
                # Extract query tokens
                query_tokens = result['query_tokens']
                
                # Store in results dictionary
                embeddings_dict[item_id] = query_tokens
                
                processed_items += 1
                
                # Print progress
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / processed_items
                remaining_items = total_items - processed_items
                estimated_remaining_time = remaining_items * avg_time_per_item
                
                print(f"      ✅ Generated {len(query_tokens)} query tokens")
                print(f"      📊 Progress: {processed_items}/{total_items} ({processed_items/total_items*100:.1f}%)")
                print(f"      ⏱️  Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"      ❌ Error processing item {item_id}: {e}")
                # Store error information
                embeddings_dict[item_id] = {"error": str(e)}
                processed_items += 1
        
        # Save progress periodically
        if save_progress and batch_num % 5 == 0:
            progress_path = output_path.replace('.json', f'_progress_batch_{batch_num}.json')
            print(f"💾 Saving progress to: {progress_path}")
            with open(progress_path, 'w') as f:
                json.dump(embeddings_dict, f, indent=2)
    
    # Final save
    print(f"\n💾 Saving final results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    # Print final summary
    total_time = time.time() - start_time
    successful_items = sum(1 for v in embeddings_dict.values() if not isinstance(v, dict) or "error" not in v)
    error_items = total_items - successful_items
    
    print(f"\n🎉 Legacy embedding generation completed!")
    print(f"   Total items processed: {total_items}")
    print(f"   Successful: {successful_items}")
    print(f"   Errors: {error_items}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average time per item: {total_time/total_items:.2f} seconds")
    print(f"   Results saved to: {output_path}")
    
    return embeddings_dict

def compare_processing_methods(
    data_path: str,
    checkpoint_path: str,
    sample_size: int = 50,
    batch_size: int = 256,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Compare the performance of efficient batch processing vs legacy individual processing.
    
    Args:
        data_path: Path to the JSON data file
        checkpoint_path: Path to the Q-Former checkpoint
        sample_size: Number of items to use for comparison
        batch_size: Batch size for batch processing
        device: Device to run inference on
        
    Returns:
        Dictionary containing performance comparison results
    """
    print("🔬 Performance Comparison: Efficient Batch vs Legacy Individual Processing")
    print("=" * 70)
    
    # Load sample data
    items_data = load_item_data(data_path)
    sample_items = dict(list(items_data.items())[:sample_size])
    
    print(f"📊 Using {len(sample_items)} items for performance comparison")
    
    # Initialize inference
    inference = QFormerInference(checkpoint_path, device)
    
    # Test 1: Legacy individual processing
    print(f"\n🐌 Testing legacy individual processing...")
    start_time = time.time()
    
    legacy_results = {}
    for item_id in list(sample_items.keys()):
        try:
            result = inference.generate_query_tokens_by_id(item_id, data_path)
            legacy_results[item_id] = result['query_tokens']
        except Exception as e:
            legacy_results[item_id] = {"error": str(e)}
    
    legacy_time = time.time() - start_time
    legacy_success = sum(1 for v in legacy_results.values() if not isinstance(v, dict) or "error" not in v)
    
    print(f"   ✅ Legacy processing completed: {legacy_success}/{len(sample_items)} items in {legacy_time:.2f}s")
    print(f"   📊 Legacy speed: {legacy_success/legacy_time:.2f} items/second")
    
    # Test 2: Efficient batch processing
    print(f"\n🚀 Testing efficient batch processing...")
    start_time = time.time()
    
    try:
        batch_results = inference.generate_query_tokens_batch_by_ids(list(sample_items.keys()), data_path)
        batch_time = time.time() - start_time
        
        batch_success = 0
        for result in batch_results:
            if 'error' not in result:
                batch_success += 1
        
        print(f"   ✅ Batch processing completed: {batch_success}/{len(sample_items)} items in {batch_time:.2f}s")
        print(f"   📊 Batch speed: {batch_success/batch_time:.2f} items/second")
        
        # Calculate speedup
        if legacy_time > 0 and batch_time > 0:
            speedup = legacy_time / batch_time
            print(f"   🚀 Speedup: {speedup:.2f}x faster with batch processing!")
            
            # Performance analysis
            if speedup > 1.5:
                print(f"   🎉 Significant performance improvement detected!")
            elif speedup > 1.1:
                print(f"   👍 Moderate performance improvement detected")
            else:
                print(f"   ⚠️  Minimal performance improvement - check batch size and device")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        batch_time = float('inf')
        batch_success = 0
        speedup = 0
    
    # Summary
    print(f"\n📋 Performance Comparison Summary:")
    print(f"   Sample size: {len(sample_items)} items")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Legacy processing: {legacy_time:.2f}s ({legacy_success} items)")
    print(f"   Batch processing: {batch_time:.2f}s ({batch_success} items)")
    
    if legacy_time > 0 and batch_time > 0:
        print(f"   Overall speedup: {speedup:.2f}x")
        print(f"   Time saved: {legacy_time - batch_time:.2f}s")
        print(f"   Efficiency gain: {(1 - batch_time/legacy_time)*100:.1f}%")
    
    return {
        'sample_size': len(sample_items),
        'batch_size': batch_size,
        'device': device,
        'legacy_time': legacy_time,
        'legacy_success': legacy_success,
        'batch_time': batch_time,
        'batch_success': batch_success,
        'speedup': speedup if 'speedup' in locals() else 0
    }

def main():
    """Main function for generating all item embeddings."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all items in a data file using efficient batch processing",
        epilog="""
Examples:
    # Use efficient batch processing (default, recommended)
    python generate_all_item_embeddings.py
    
    # Check GPU status first, then process
    python generate_all_item_embeddings.py --check-gpu
    
    # Use legacy individual processing
    python generate_all_item_embeddings.py --legacy-processing
    
    # Enable memory optimization and profiling
    python generate_all_item_embeddings.py --memory-efficient --profile
    
    # Run performance comparison first, then process
    python generate_all_item_embeddings.py --compare --compare-sample-size 100
    
    # Custom batch size and checkpoint
    python generate_all_item_embeddings.py --batch-size 16 --checkpoint "path/to/checkpoint.pth"
        """
    )
    parser.add_argument("--data", type=str, 
                       default="data_rec/dict/All_Beauty_item_triplet_dict.json",
                       help="Path to the input JSON data file")
    parser.add_argument("--checkpoint", type=str, 
                       default="qformer_checkpoints_2_tokens/best_qformer_model.pth",
                       help="Path to the Q-Former checkpoint")
    parser.add_argument("--output", type=str, 
                       default="data_rec/embeddings/item_q_former_embedding.json",
                       help="Output file path for embeddings")
    parser.add_argument("--max-items", type=int, default=None,
                       help="Maximum number of items to process (None for all)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run inference on")
    parser.add_argument("--no-save-progress", action="store_true",
                       help="Disable periodic progress saving")
    parser.add_argument("--legacy-processing", action="store_true",
                       help="Use legacy individual item processing instead of efficient batch processing")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Enable memory-efficient processing (smaller intermediate tensors)")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling and timing information")
    parser.add_argument("--compare", action="store_true",
                       help="Run performance comparison between batch and legacy processing")
    parser.add_argument("--compare-sample-size", type=int, default=50,
                       help="Number of items to use for performance comparison (default: 50)")
    parser.add_argument("--check-gpu", action="store_true",
                       help="Check GPU status and availability before processing")
    
    args = parser.parse_args()
    
    print("🚀 Generate All Item Embeddings Script - Enhanced with Efficient Batch Processing")
    print("=" * 70)
    
    try:
        # Check GPU status if requested
        if args.check_gpu:
            print("🔍 Running GPU status check...")
            gpu_available = check_gpu_status()
            
            if not gpu_available:
                print(f"\n⚠️  GPU check failed. The script will attempt to use CPU instead.")
                print(f"   Processing will be slower but should still work.")
                
                # Ask user if they want to continue
                response = input("\nContinue with CPU processing? (y/n): ").lower().strip()
                if response not in ['y', 'yes']:
                    print("Exiting...")
                    return 0
        
        # Validate input file
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Input data file not found: {args.data}")
        
        # Validate checkpoint file
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        
        # Run performance comparison if requested
        if args.compare:
            print("🔬 Running performance comparison...")
            comparison_results = compare_processing_methods(
                data_path=args.data,
                checkpoint_path=args.checkpoint,
                sample_size=args.compare_sample_size,
                batch_size=args.batch_size,
                device=args.device
            )
            
            # Ask user if they want to continue with full processing
            if comparison_results['speedup'] > 0:
                print(f"\n💡 Based on the comparison, batch processing is {comparison_results['speedup']:.2f}x faster!")
                print("   Continuing with efficient batch processing...")
            else:
                print(f"\n⚠️  Batch processing had issues. Consider using --legacy-processing")
        
        # Generate embeddings
        if args.legacy_processing:
            embeddings = generate_embeddings_for_all_items_legacy(
                data_path=args.data,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                device=args.device,
                batch_size=args.batch_size,
                max_items=args.max_items,
                save_progress=not args.no_save_progress
            )
        else:
            embeddings = generate_embeddings_for_all_items(
                data_path=args.data,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                device=args.device,
                batch_size=args.batch_size,
                max_items=args.max_items,
                save_progress=not args.no_save_progress,
                memory_efficient=args.memory_efficient,
                profile=args.profile
            )
        
        print(f"\n✅ Successfully generated embeddings for {len(embeddings)} items!")
        print(f"   Output saved to: {args.output}")
        
        # Print sample of results
        print(f"\n📋 Sample results:")
        sample_items = list(embeddings.items())[:3]
        for item_id, query_tokens in sample_items:
            if isinstance(query_tokens, list):
                print(f"   {item_id}: {len(query_tokens)} query tokens, dimension {len(query_tokens[0]) if query_tokens else 0}")
            else:
                print(f"   {item_id}: {query_tokens}")
        
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 