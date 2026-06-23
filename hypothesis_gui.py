import json
import sys
import os

# ANSI escape codes for terminal coloring
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def colorize_level(level_str):
    """Adds color to HIGH/MEDIUM/LOW strings"""
    if not level_str:
        return "N/A"
    level = str(level_str).upper()
    if level == "HIGH":
        return f"{Colors.GREEN}{level}{Colors.RESET}"
    elif level == "MEDIUM":
        return f"{Colors.YELLOW}{level}{Colors.RESET}"
    elif level == "LOW":
        return f"{Colors.RED}{level}{Colors.RESET}"
    return level

def main():
    file_path = "ai_hypotheses.json"
    
    if not os.path.exists(file_path):
        print(f"{Colors.RED}Error: Could not find '{file_path}' in the current directory.{Colors.RESET}")
        return

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"{Colors.RED}Error: Failed to parse JSON file.{Colors.RESET}")
            return

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*50}")
    print(f"       AI HYPOTHESIS SEARCH RESULTS")
    print(f"{'='*50}{Colors.RESET}\n")

    # --- Print Meta Overview ---
    overview = data.get('final_overview', {})
    print(f"{Colors.CYAN}{Colors.BOLD}[ META SUMMARY ]{Colors.RESET}")
    print(overview.get('summary', 'No summary available.') + "\n")

    # --- Print Ranked Hypotheses ---
    ranked = data.get('hypotheses_ranked', [])
    print(f"{Colors.YELLOW}{Colors.BOLD}[ TOP RANKED HYPOTHESES ]{Colors.RESET}")
    
    if not ranked:
        print("No hypotheses found.")
        return

    for i, hyp in enumerate(ranked):
        # Extract basic data
        h_id = hyp.get('id', 'Unknown ID')
        title = hyp.get('title', 'Untitled')
        elo = hyp.get('elo_score', 0)
        novelty = hyp.get('novelty_review')
        feasibility = hyp.get('feasibility_review')
        
        # Extract verification evidence safely
        evidence = hyp.get('verification', {}).get('evidence', {})
        chern = evidence.get('chern_number', {}).get('chern', 'N/A')
        bws = evidence.get('bandwidths', [])
        
        # Format bandwidths beautifully
        min_bw = min(bws) if bws else None
        bw_str = f"{min_bw:.3f}t (min)" if min_bw else "N/A"

        print(f"{Colors.BOLD}{i+1}. [{h_id}] {title}{Colors.RESET}")
        print(f"   {Colors.BLUE}Elo Score:{Colors.RESET}   {elo:.1f}")
        print(f"   {Colors.BLUE}Reviews:{Colors.RESET}     Novelty: {colorize_level(novelty)} | Feasibility: {colorize_level(feasibility)}")
        print(f"   {Colors.BLUE}Chern #:{Colors.RESET}     {chern}")
        print(f"   {Colors.BLUE}Bandwidth:{Colors.RESET}   {bw_str}")
        print("-" * 50)

if __name__ == "__main__":
    main()