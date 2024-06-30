from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)

def log_summary(results):
    total_attacks = len(results)
    if total_attacks == 0:
        return

    # Default metrics - calculated on every attack
    attack_success_stats = AttackSuccessRate().calculate(results)
    words_perturbed_stats = WordsPerturbed().calculate(results)
    attack_query_stats = AttackQueries().calculate(results)

    # @TODO generate this table based on user input - each column in specific class
    # Example to demonstrate:
    # summary_table_rows = attack_success_stats.display_row() + words_perturbed_stats.display_row() + ...
    summary_table_rows = [
        [
            "Number of successful attacks:",
            attack_success_stats["successful_attacks"],
        ],
        ["Number of failed attacks:", attack_success_stats["failed_attacks"]],
        ["Number of skipped attacks:", attack_success_stats["skipped_attacks"]],
        [
            "Original accuracy:",
            str(attack_success_stats["original_accuracy"]) + "%",
        ],
        [
            "Accuracy under attack:",
            str(attack_success_stats["attack_accuracy_perc"]) + "%",
        ],
        [
            "Attack success rate:",
            str(attack_success_stats["attack_success_rate"]) + "%",
        ],
        [
            "Average perturbed word %:",
            str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%",
        ],
        [
            "Average num. words per input:",
            words_perturbed_stats["avg_word_perturbed"],
        ],
    ]

    summary_table_rows.append(
        ["Avg num queries:", attack_query_stats["avg_num_queries"]]
    )

    for metric_name, metric in self.metrics.items():
        summary_table_rows.append([metric_name, metric.calculate(self.results)])

    self.log_summary_rows(
        summary_table_rows, "Attack Results", "attack_results_summary"
    )
