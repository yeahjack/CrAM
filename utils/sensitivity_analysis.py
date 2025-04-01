import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from re_weighting import *

# Assuming you've imported the Re_Weighting_Strategy and Find_Best_Heads classes from your original code

# Complex question about climate science
question = "Explain the relationship between methane emissions, the carbon cycle, and global warming, including the comparative impact of different greenhouse gases over various time horizons."

# Longer, more detailed paragraphs with varying relevance
paras = [
    # Para 1: Highly relevant - about methane and its comparative impact
    "Methane (CH₄) is a potent greenhouse gas with a global warming potential (GWP) approximately 28-36 times that of carbon dioxide (CO₂) over a 100-year period, but its impact is even more significant in the short term, with a GWP of 84-87 over a 20-year horizon. This is primarily because methane has a much shorter atmospheric lifetime of about 12 years compared to CO₂, which can persist for centuries. Methane emissions originate from both natural sources (wetlands, termites, and geological seeps) and anthropogenic activities (agriculture, particularly rice cultivation and livestock farming, fossil fuel extraction and distribution, and waste management). Despite its shorter lifetime, methane contributes significantly to global warming due to its efficiency in trapping infrared radiation. Recent research indicates that methane is responsible for approximately 30% of the global temperature rise since pre-industrial times, making it the second most important greenhouse gas after carbon dioxide in terms of current climate forcing. The rapid increase in atmospheric methane concentrations observed since 2007, after a period of relative stability, has raised concerns among climate scientists about potential feedback loops in the carbon cycle.",
    
    # Para 2: Moderately relevant - about the carbon cycle
    "The carbon cycle represents the continuous exchange of carbon between the atmosphere, oceans, soil, and living organisms through processes like photosynthesis, respiration, decomposition, and combustion. In natural conditions, this cycle maintains a balance that has kept atmospheric greenhouse gas concentrations relatively stable over millennia. However, human activities have significantly disrupted this equilibrium primarily through fossil fuel combustion, deforestation, and industrial processes, releasing carbon that was previously sequestered for millions of years. Oceanic and terrestrial carbon sinks currently absorb approximately 50% of anthropogenic CO₂ emissions, with the oceans absorbing about 25% and terrestrial ecosystems like forests absorbing another 25%. However, the efficiency of these natural carbon sinks is declining as they approach saturation points and as climate change itself affects ecosystem functioning. For instance, warming oceans can absorb less CO₂, while increasing wildfires and thawing permafrost release additional carbon into the atmosphere. These feedback mechanisms are particularly concerning because they can accelerate warming beyond direct anthropogenic emissions, potentially leading to tipping points in the climate system.",
    
    # Para 3: Highly relevant - about comparative greenhouse gas impacts
    "While carbon dioxide (CO₂) is the most abundant greenhouse gas and the largest contributor to anthropogenic climate change in terms of total forcing, other greenhouse gases have significantly higher warming potentials per molecule. Nitrous oxide (N₂O), primarily from agricultural fertilizers and industrial processes, has a global warming potential approximately 265-298 times that of CO₂ over a 100-year period and an atmospheric lifetime of about 114 years. Fluorinated gases such as hydrofluorocarbons (HFCs), perfluorocarbons (PFCs), and sulfur hexafluoride (SF₆) have even higher global warming potentials, ranging from thousands to tens of thousands times that of CO₂, and extremely long atmospheric lifetimes. The time horizon used for comparison significantly affects how we prioritize mitigation efforts. When using a 20-year horizon instead of the more common 100-year horizon, the relative importance of short-lived climate pollutants like methane increases substantially. This temporal aspect of climate forcing has important policy implications, as reducing methane emissions could provide significant near-term climate benefits while longer-term strategies focus on CO₂ reductions. The Intergovernmental Panel on Climate Change (IPCC) has emphasized that limiting global warming to 1.5°C or 2°C above pre-industrial levels requires rapid reductions in all greenhouse gas emissions, not just carbon dioxide.",
    
    # Para 4: Less relevant - about renewable energy technologies
    "Renewable energy technologies have advanced significantly in recent decades, with solar photovoltaic and wind power now representing the cheapest forms of new electricity generation in many regions worldwide. The levelized cost of electricity (LCOE) from utility-scale solar PV decreased by approximately 85% between 2010 and 2020, while onshore wind LCOE fell by about 56% during the same period. These dramatic cost reductions have accelerated deployment, with global renewable capacity additions reaching record levels despite the COVID-19 pandemic. Energy storage technologies, particularly lithium-ion batteries, have followed similar cost reduction curves, addressing the intermittency challenges associated with variable renewable sources. Beyond electricity generation, electrification of transportation, heating, and industrial processes represents a critical pathway for decarbonization when coupled with renewable electricity. However, certain sectors remain difficult to electrify, including aviation, shipping, and high-temperature industrial processes, necessitating the development of sustainable biofuels, green hydrogen, and synthetic fuels. The transition to renewable energy systems provides co-benefits beyond climate change mitigation, including reduced air pollution, enhanced energy security, distributed economic benefits, and resilience against fuel price volatility.",
    
    # Para 5: Moderately relevant - about feedback loops in climate systems
    "Climate feedback mechanisms represent processes that either amplify (positive feedback) or diminish (negative feedback) the effects of climate forcings. Several positive feedback loops in the Earth system are particularly concerning for their potential to accelerate global warming beyond direct anthropogenic emissions. Arctic amplification, where the Arctic warms faster than the global average due to decreasing ice cover and reduced albedo, represents one such feedback. As sea ice and snow cover diminish, darker land and ocean surfaces absorb more solar radiation, further increasing regional warming. Similarly, thawing permafrost in Arctic and boreal regions releases previously frozen methane and carbon dioxide, adding to atmospheric greenhouse gas concentrations. Ocean warming reduces the solubility of gases, leading to less carbon dioxide absorption and potentially increased release from the oceans. Water vapor feedback is perhaps the most significant amplifying mechanism, as rising temperatures increase atmospheric water vapor content, which itself is a potent greenhouse gas. Cloud feedbacks remain a significant source of uncertainty in climate projections, as different cloud types and altitudes can either enhance or reduce warming depending on their properties. These interconnected feedback mechanisms create the potential for cascading effects and non-linear changes in the climate system that are difficult to predict with precision."
]

# Relevance scores for the paragraphs (higher means more relevant)
scores = [0.92, 0.75, 0.88, 0.45, 0.70]

# Multiple pairs of right and wrong answers
answer_pairs = [
    # Pair 1: About methane's warming potential
    {
        "right_answer": "Methane has a global warming potential 28-36 times that of CO₂ over 100 years but 84-87 times over 20 years due to its shorter atmospheric lifetime of about 12 years. Despite this shorter lifetime, methane contributes significantly to global warming because it is highly effective at trapping infrared radiation and is responsible for approximately 30% of global temperature rise since pre-industrial times.",
        "wrong_answer": "Methane has a global warming potential about 5-10 times that of CO₂ and has a long atmospheric lifetime of about 100 years, making it the most persistent greenhouse gas. Its contribution to global warming is minimal compared to CO₂, accounting for less than 5% of temperature increases."
    },
    
    # Pair 2: About the carbon cycle
    {
        "right_answer": "The carbon cycle involves continuous exchange between atmosphere, oceans, soil, and living organisms. Human activities have disrupted this balance through fossil fuel combustion and deforestation. Currently, natural sinks absorb about 50% of anthropogenic CO₂ emissions, but their efficiency is declining as they approach saturation and as climate change affects ecosystem functioning.",
        "wrong_answer": "The carbon cycle is primarily driven by volcanic eruptions and natural geological processes, with minimal human influence. Natural carbon sinks like oceans and forests easily compensate for all human emissions, absorbing nearly 100% of anthropogenic CO₂, with no risk of saturation or declining efficiency."
    },
    
    # Pair 3: About greenhouse gas comparisons and time horizons
    {
        "right_answer": "Different greenhouse gases have varying warming potentials and atmospheric lifetimes, affecting how we prioritize mitigation. While CO₂ is the largest contributor to climate change overall, methane, nitrous oxide, and fluorinated gases have much higher warming potentials per molecule. Using a 20-year time horizon instead of 100 years significantly increases the relative importance of short-lived pollutants like methane.",
        "wrong_answer": "All greenhouse gases contribute equally to global warming regardless of their molecular structure. The time horizon used for comparison has no effect on mitigation priorities, and all gases should be treated identically in climate policy, with no special consideration for short-lived versus long-lived climate pollutants."
    }
]

def run_sensitivity_analysis(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """Run sensitivity analysis with multiple wrong answers to see their effect on head identification"""
    best_heads_finder = Find_Best_Heads(model_name=model_name)
    
    # Store results for each answer pair
    all_results = []
    
    print(f"\nRunning sensitivity analysis with {len(answer_pairs)} different right/wrong answer pairs...")
    
    for i, pair in enumerate(answer_pairs):
        print(f"\n----- Testing Answer Pair {i+1} -----")
        print(f"RIGHT: {pair['right_answer'][:100]}...")
        print(f"WRONG: {pair['wrong_answer'][:100]}...")
        
        start_time = time.time()
        prob_changes = best_heads_finder.cal_logits(
            question=question, 
            paras=paras, 
            scores=scores, 
            right_answer=pair['right_answer'], 
            wrong_answer=pair['wrong_answer']
        )
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        
        # Find top heads
        flat_changes = [(layer, head, change) for layer, layer_changes in enumerate(prob_changes) 
                       for head, change in enumerate(layer_changes)]
        top_heads = sorted(flat_changes, key=lambda x: x[2], reverse=True)[:10]
        
        print("\nTop 10 most influential attention heads for this answer pair:")
        for layer, head, change in top_heads[:10]:
            print(f"Layer: {layer}, Head: {head}, Change: {change:.4f}")
        
        all_results.append({
            "pair_index": i,
            "top_heads": top_heads,
            "prob_changes": prob_changes
        })
    
    return all_results

def compare_results(all_results):
    """Compare the results from different answer pairs"""
    print("\n===== Comparing Results Across Answer Pairs =====")
    
    # Extract top 5 heads from each pair
    top_heads_by_pair = [[(layer, head) for layer, head, _ in result["top_heads"][:5]] 
                         for result in all_results]
    
    # Find heads that appear in multiple pairs
    all_top_heads = [head for pair_heads in top_heads_by_pair for head in pair_heads]
    head_counts = {}
    for head in all_top_heads:
        if head in head_counts:
            head_counts[head] += 1
        else:
            head_counts[head] = 1
    
    common_heads = {head: count for head, count in head_counts.items() if count > 1}
    print(f"\nHeads appearing in top 5 for multiple answer pairs:")
    for (layer, head), count in sorted(common_heads.items(), key=lambda x: x[1], reverse=True):
        print(f"Layer {layer}, Head {head}: Appears in {count} answer pairs")
    
    # Visualize the comparison
    plot_head_comparison(all_results)
    
    return common_heads

def plot_head_comparison(all_results):
    """Create a heatmap visualization of head importance across answer pairs"""
    # This is commented out but would create visualizations if implemented
    # You would need matplotlib and seaborn installed
    """
    num_layers = max([layer for result in all_results for layer, _, _ in result["top_heads"]]) + 1
    num_heads = max([head for result in all_results for _, head, _ in result["top_heads"]]) + 1
    
    fig, axes = plt.subplots(len(all_results), 1, figsize=(12, 4*len(all_results)), sharex=True)
    
    for i, result in enumerate(all_results):
        data = np.zeros((num_layers, num_heads))
        for layer, head, change in result["top_heads"][:20]:  # Top 20 for visualization
            data[layer, head] = change
        
        ax = axes[i] if len(all_results) > 1 else axes
        sns.heatmap(data, ax=ax, cmap="YlOrRd", 
                   xticklabels=range(num_heads), 
                   yticklabels=range(num_layers))
        ax.set_title(f"Answer Pair {i+1}: Attention Head Importance")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
    
    plt.tight_layout()
    plt.savefig("attention_head_comparison.png")
    print("Visualization saved as 'attention_head_comparison.png'")
    """
    pass

def run_with_best_heads(model_name="meta-llama/Llama-2-7b-chat-hf", common_heads=None):
    """Run the RAG system with the identified best heads"""
    if not common_heads:
        # If no common heads provided, use some defaults for demonstration
        layers_to_be_modified = {0: [5], 10: [3], 20: [7]}
    else:
        # Use the heads that appear in multiple answer pairs
        layers_to_be_modified = {layer: [head] for (layer, head), _ in 
                                sorted(common_heads.items(), key=lambda x: x[1], reverse=True)[:5]}
    
    print(f"\nRunning RAG with selected attention heads: {layers_to_be_modified}")
    
    # Initialize the RAG strategy with the selected heads
    rag_strategy = Re_Weighting_Strategy(
        model_name=model_name, 
        layers_to_be_modified=layers_to_be_modified
    )
    
    # Run the generation
    start_time = time.time()
    prompt, output = rag_strategy.run_RAG_with_attention_weighting(
        question=question, paras=paras, scores=scores
    )
    end_time = time.time()
    
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print(f"Final answer: {output}")
    
    return output

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change to your preferred model
    
    # Run the basic RAG process with default heads (as baseline)
    print("\n===== BASELINE: Using Default Attention Heads =====")
    rag_strategy = Re_Weighting_Strategy(model_name=model_name)
    start_time = time.time()
    prompt, baseline_output = rag_strategy.run_RAG_with_attention_weighting(
        question=question, paras=paras, scores=scores
    )
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print(f"Baseline answer: {baseline_output}")
    
    # Run sensitivity analysis with different answer pairs
    print("\n===== ANALYSIS: Finding Best Attention Heads =====")
    results = run_sensitivity_analysis(model_name)
    
    # Compare results across different answer pairs
    common_heads = compare_results(results)
    
    # Run with identified best heads
    print("\n===== OPTIMIZED: Using Identified Best Heads =====")
    optimized_output = run_with_best_heads(model_name, common_heads)
    
    # Final comparison
    print("\n===== RESULTS COMPARISON =====")
    print(f"BASELINE: {baseline_output}")
    print(f"OPTIMIZED: {optimized_output}")