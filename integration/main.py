#!/usr/bin/env python
"""
Traffic Optimization Integration Layer - Main Entry Point

This script provides a CLI interface to run the integrated traffic
optimization pipeline.

Usage:
    python -m integration.main --input data.csv --mode batch
    python -m integration.main --demo
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Traffic Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m integration.main --demo
  python -m integration.main --input outputs/module1_results.csv
  python -m integration.main --input data.csv --mode batch --output results.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file with traffic data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output file for results'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['single', 'batch'],
        default='single',
        help='Processing mode: single (last sample) or batch (all samples)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a demonstration with sample data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-lstm',
        action='store_true',
        help='Disable LSTM prediction'
    )
    
    parser.add_argument(
        '--no-vae',
        action='store_true',
        help='Disable VAE anomaly detection'
    )
    
    parser.add_argument(
        '--no-rl',
        action='store_true',
        help='Disable RL optimization'
    )
    
    parser.add_argument(
        '--enable-gan',
        action='store_true',
        help='Enable GAN data augmentation'
    )
    
    return parser.parse_args()


def run_demo():
    """Run a demonstration of the pipeline."""
    print("=" * 60)
    print("Traffic Optimization Pipeline - Demo")
    print("=" * 60)
    print()
    
    from .config import Config
    from .pipeline import TrafficOptimizationPipeline, PipelineOptions
    from .data_pipeline import DataPipeline
    
    # Initialize with default config
    config = Config()
    options = PipelineOptions(
        enable_vision=False,  # Skip vision for demo (needs images)
        enable_lstm=True,
        enable_vae=True,
        enable_gan=False,
        enable_rl=True,
        vae_epochs=30  # Fewer epochs for demo
    )
    
    print("1. Initializing pipeline...")
    pipeline = TrafficOptimizationPipeline(config=config, options=options)
    
    # Check if default data file exists
    default_data = config.get_data_path("module1_results.csv")
    
    if default_data.exists():
        print(f"2. Loading data from {default_data}...")
        pipeline.load_data(default_data)
    else:
        print("2. Creating sample data...")
        # Create sample data for demo
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'image_name': [f'sample_{i}.jpg' for i in range(n_samples)],
            'vehicle_count': np.random.poisson(12, n_samples),
            'Pedestrian_count': np.random.poisson(2, n_samples),
            'congestion': ['LOW' if v < 10 else 'Medium' if v < 25 else 'High' 
                          for v in np.random.poisson(12, n_samples)]
        })
        
        pipeline.data_pipeline = DataPipeline(config)
        pipeline.data_pipeline.load_from_dataframe(sample_data)
        print(f"   Created {n_samples} sample data points")
    
    print(f"3. Running pipeline on {len(pipeline.data_pipeline)} samples...")
    print()
    
    # Run the pipeline
    result = pipeline.run()
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(result.summary())
    print()
    
    # Show RL policy summary
    if pipeline._rl_wrapper:
        print()
        print("Traffic Signal Policy:")
        print("-" * 40)
        print(pipeline._rl_wrapper.get_policy_summary())
    
    print()
    print("Demo completed successfully!")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.demo:
        run_demo()
        return 0
    
    # Import here to avoid slow startup for --help
    from .config import Config
    from .pipeline import TrafficOptimizationPipeline, PipelineOptions
    
    # Configure options from arguments
    options = PipelineOptions(
        enable_vision=False,  # Vision requires images, not CSV
        enable_lstm=not args.no_lstm,
        enable_vae=not args.no_vae,
        enable_gan=args.enable_gan,
        enable_rl=not args.no_rl
    )
    
    # Initialize pipeline
    config = Config()
    pipeline = TrafficOptimizationPipeline(config=config, options=options)
    
    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = config.get_data_path("module1_results.csv")
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Use --demo to run with sample data, or specify --input")
        return 1
    
    # Load data
    print(f"Loading data from {input_path}...")
    pipeline.load_data(input_path)
    print(f"Loaded {len(pipeline.data_pipeline)} samples")
    
    # Run pipeline
    if args.mode == 'batch':
        print("Running batch processing...")
        results = pipeline.run_batch()
        print(f"Processed {len(results)} samples")
        
        # Save results if output specified
        if args.output:
            import json
            output_data = [r.to_dict() for r in results]
            
            output_path = Path(args.output)
            if output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
            else:
                # CSV output
                import pandas as pd
                df = pd.DataFrame([
                    {
                        'action': r.action.action_name if r.action else None,
                        'is_anomaly': r.anomaly.is_anomaly if r.anomaly else None,
                    }
                    for r in results
                ])
                df.to_csv(output_path, index=False)
            
            print(f"Results saved to {output_path}")
    else:
        print("Running single analysis...")
        result = pipeline.run()
        print()
        print(result.summary())
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"Result saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
