## User Interface Requirements

### Control Panel (Policy Levers)

Users adjust:
1. **Congestion threshold:** Target classroom-to-student ratio (25:1 to 50:1)
2. **Rank perturbation tolerance:** Max % students moved to Rank 2/3 (0% to 30%)
3. **ESC subsidy amounts:** By tier — NCR (₱13k), Region IV-A HUC/Lucena (₱11k), Region IV-A other (₱9k)
4. **Slot budget:** Total ESC slots available, split by NCR and Region IV-A

### Visualization Outputs

1. **Summary Cards:** Students affected, congestion relief, budget utilization, preference respect
2. **Scenario Comparison Table:** Side-by-side baseline vs. scenarios
3. **Flow Visualization:** Sankey diagram showing student flows from congested public → ESC private
4. **Geographic Heatmap:** Congestion relief by municipality within NCR and Region IV-A (optional)

### Design Principles

1. **Progressive Disclosure:** Simple sliders upfront, advanced options behind toggle
2. **Instant Feedback:** Slider adjustments update preview before "Run Simulation"
3. **Transparency:** Always show objective function and constraints
4. **Comparative Framing:** Never show scenario in isolation; always compare to baseline
5. **Non-Dictatorial Tone:** "Suggested allocation" not "Required allocation"

## Edge Cases & Validation

### Infeasible Scenarios

**Trigger:** Slot budget < students needing reassignment

**Display:** "Cannot achieve 30:1 ratio with current slot budget. Options: (a) Increase slots to X, (b) Relax threshold to Y:1"

### Unused Slots

**Trigger:** ESC slots remain unfilled after optimization

**Display:** "120 slots unused. Why? (a) Distance too far (avg 8km), (b) Tuition gap still high (₱5k), (c) Families unaware"