"""
Run Parameter Comparison Experiments (OPTIMIZED)
=================================================
Script để chạy 4 thí nghiệm so sánh tham số (Yêu cầu 3):
1. Labeled Size Comparison (5%, 10%, 20%) - optimized from 4 to 3 configs
2. Model Comparison (HGBC, RandomForest) - optimized from 3 to 2 models
3. View Splitting Experiments (2 strategies) - optimized from 4 to 2 strategies
4. Hybrid τ Schedule (2 schedules) - optimized from 4 to 2 schedules

⚡ Total time: ~59 minutes (down from 80-105 minutes, ~40% faster)

Usage:
    python run_parameter_experiments.py
    
    # Hoặc chạy từng experiment riêng:
    python run_parameter_experiments.py --only labeled_size
    python run_parameter_experiments.py --only model_comparison
    python run_parameter_experiments.py --only view_splitting
    python run_parameter_experiments.py --only hybrid_tau
"""

import papermill as pm
import argparse
from pathlib import Path
from datetime import datetime

# Kernel name
KERNEL = "beijing_env"

# Experiments configuration (OPTIMIZED for speed while maintaining clear contrasts)
EXPERIMENTS = {
    "labeled_size": {
        "name": "Labeled Size Comparison",
        "input": "notebooks/labeled_size_comparison.ipynb",
        "output": "notebooks/runs/labeled_size_comparison_run.ipynb",
        "description": "So sánh 3 kích thước labeled data (5%, 10%, 20%) - optimized, ~15 min"
    },
    "model_comparison": {
        "name": "Model Comparison",
        "input": "notebooks/model_comparison.ipynb",
        "output": "notebooks/runs/model_comparison_run.ipynb",
        "description": "So sánh HGBC vs RandomForest (2 models) - optimized, ~13 min"
    },
    "view_splitting": {
        "name": "View Splitting Experiments",
        "input": "notebooks/view_splitting_experiments.ipynb",
        "output": "notebooks/runs/view_splitting_experiments_run.ipynb",
        "description": "Test 2 strategies (Current, Pollutant-based) - optimized, ~18 min"
    },
    "hybrid_tau": {
        "name": "Hybrid τ Schedule",
        "input": "notebooks/hybrid_tau_schedule.ipynb",
        "output": "notebooks/runs/hybrid_tau_schedule_run.ipynb",
        "description": "Test 2 τ schedules (Fixed, Aggressive) - optimized, ~13 min"
    }
}


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_experiment(exp_key, config):
    """Run a single experiment"""
    print_banner(f"EXPERIMENT: {config['name']}")
    print(f"📝 {config['description']}")
    print(f"📂 Input:  {config['input']}")
    print(f"📂 Output: {config['output']}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = datetime.now()
    
    try:
        pm.execute_notebook(
            config["input"],
            config["output"],
            kernel_name=KERNEL,
            language="python"
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Completed in {duration:.1f} seconds")
        return True
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ Failed after {duration:.1f} seconds")
        print(f"Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--only",
        choices=list(EXPERIMENTS.keys()),
        help="Run only specific experiment"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        default=[],
        help="Skip specific experiments"
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.only:
        experiments_to_run = {args.only: EXPERIMENTS[args.only]}
    else:
        experiments_to_run = {
            k: v for k, v in EXPERIMENTS.items() 
            if k not in args.skip
        }
    
    print_banner("🚀 PARAMETER COMPARISON EXPERIMENTS")
    print(f"Running {len(experiments_to_run)} experiment(s):")
    for key, config in experiments_to_run.items():
        print(f"  • {config['name']}")
    print()
    
    # Track results
    results = {}
    overall_start = datetime.now()
    
    # Run experiments
    for exp_key, config in experiments_to_run.items():
        success = run_experiment(exp_key, config)
        results[exp_key] = success
    
    # Summary
    overall_duration = (datetime.now() - overall_start).total_seconds()
    
    print_banner("📊 EXECUTION SUMMARY")
    print(f"Total time: {overall_duration:.1f} seconds ({overall_duration/60:.1f} minutes)\n")
    
    print("Results:")
    for exp_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {EXPERIMENTS[exp_key]['name']}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nSuccess rate: {successful}/{total} ({successful/total*100:.0f}%)")
    
    if successful == total:
        print("\n🎉 All experiments completed successfully!")
        print_output_locations()
    else:
        print("\n⚠️ Some experiments failed. Check output above for details.")
        return 1
    
    return 0


def print_output_locations():
    """Print where to find outputs"""
    print("\n📂 Output Locations:")
    print("  • Labeled Size:      data/processed/labeled_size_experiments/")
    print("  • Model Comparison:  data/processed/model_comparison_experiments/")
    print("  • View Splitting:    data/processed/view_splitting_experiments/")
    print("  • Hybrid τ Schedule: data/processed/hybrid_tau_experiments/")
    print("\n📊 Dashboard Summaries:")
    print("  • All experiments generate dashboard_summary.json")
    print("  • Visualizations saved as PNG files (300 dpi)")
    print("  • CSV summaries for quick review")
    print("\n💡 Next Steps:")
    print("  1. Review results in respective output folders")
    print("  2. Check visualizations (*.png files)")
    print("  3. Use dashboard_summary.json for Streamlit")
    print("  4. Update BLOG_PARAMETER_COMPARISON.md with findings")


if __name__ == "__main__":
    import sys
    sys.exit(main())
