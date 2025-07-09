import os
import re

def compare_model_results():
    """
    Read all result files and create a summary comparison
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found. Run the models first.")
        return
    
    summary = []
    summary.append("="*60)
    summary.append("MODEL PERFORMANCE COMPARISON SUMMARY")
    summary.append("="*60)
    summary.append("")
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    
    for file in result_files:
        file_path = os.path.join(results_dir, file)
        summary.append(f"📁 {file}")
        summary.append("-" * 40)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract F1 scores using regex
            f1_scores = re.findall(r'F1[^:]*:\s*([\d.]+)', content)
            if f1_scores:
                summary.append(f"F1 Scores found: {', '.join(f1_scores)}")
            
            # Extract model names and performance
            sections = content.split("===")
            for section in sections[1:]:  # Skip first empty section
                lines = section.strip().split('\n')
                if lines:
                    model_name = lines[0].strip()
                    # Look for key metrics in this section
                    section_text = '\n'.join(lines)
                    precision_match = re.search(r'weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', section_text)
                    if precision_match:
                        precision, recall, f1 = precision_match.groups()
                        summary.append(f"  {model_name}: P={precision} R={recall} F1={f1}")
            
        except Exception as e:
            summary.append(f"Error reading {file}: {e}")
        
        summary.append("")
    
    # Save summary
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print("Comparison summary saved to results/comparison_summary.txt")
    print("\nQuick overview:")
    for line in summary[:20]:  # Show first 20 lines
        print(line)

if __name__ == "__main__":
    compare_model_results()
