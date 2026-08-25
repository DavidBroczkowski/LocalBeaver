import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "experiments/sort/data/sort_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(position, token):
        if position in {0, 3, 6, 7}:
            return token == "3"
        elif position in {1}:
            return token == "0"
        elif position in {2, 4, 5}:
            return token == "4"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 7}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 2, 3}:
            return token == "3"
        elif position in {1, 7}:
            return token == "0"
        elif position in {4, 5}:
            return token == "4"
        elif position in {6}:
            return token == "2"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 7}:
            return k_position == 3
        elif q_position in {2, 4}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 2

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 5, 6}:
            return token == "0"
        elif position in {1, 7}:
            return token == "2"
        elif position in {2, 3, 4}:
            return token == "1"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 7}:
            return token == "1"
        elif position in {1, 2, 3}:
            return token == "0"
        elif position in {4}:
            return token == "</s>"
        elif position in {5, 6}:
            return token == "4"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3}:
            return token == "<s>"
        elif position in {4, 5, 6}:
            return token == "3"
        elif position in {7}:
            return token == "1"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2}:
            return token == "1"
        elif position in {3, 4, 5, 7}:
            return token == "0"
        elif position in {6}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {5, 6}:
            return 3
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {6, 7}:
            return 1
        elif key in {4, 5}:
            return 5
        return 7

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output, num_attn_0_0_output):
        key = (num_attn_0_3_output, num_attn_0_0_output)
        if key in {
            (np.int64(0), np.int64(1)),
            (np.int64(0), np.int64(2)),
            (np.int64(1), np.int64(0)),
            (np.int64(1), np.int64(1)),
            (np.int64(2), np.int64(0)),
        }:
            return 1
        elif key in {(np.int64(0), np.int64(0))}:
            return 6
        return 3

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {np.int64(0), np.int64(1)}:
            return 5
        return 1

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 3, 4, 7}:
            return token == "3"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "0"
        elif position in {5, 6}:
            return token == "4"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {6, 7}:
            return k_position == 2

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0}:
            return token == "<pad>"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "2"
        elif position in {3, 6, 7}:
            return token == "1"
        elif position in {4}:
            return token == "<s>"
        elif position in {5}:
            return token == "3"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, attn_0_1_output):
        if mlp_0_0_output in {0, 3, 6}:
            return attn_0_1_output == "2"
        elif mlp_0_0_output in {1, 4, 7}:
            return attn_0_1_output == "0"
        elif mlp_0_0_output in {2}:
            return attn_0_1_output == "1"
        elif mlp_0_0_output in {5}:
            return attn_0_1_output == "3"

    num_attn_1_0_pattern = select(attn_0_1_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_3_output):
        if position in {0}:
            return attn_0_3_output == "2"
        elif position in {1}:
            return attn_0_3_output == "3"
        elif position in {2, 3, 7}:
            return attn_0_3_output == "1"
        elif position in {4}:
            return attn_0_3_output == "<pad>"
        elif position in {5}:
            return attn_0_3_output == "</s>"
        elif position in {6}:
            return attn_0_3_output == "4"

    num_attn_1_1_pattern = select(attn_0_3_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2, 3, 4}:
            return token == "0"
        elif position in {5, 7}:
            return token == "1"
        elif position in {6}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 7}:
            return token == "3"
        elif position in {2, 3, 4}:
            return token == "2"
        elif position in {5, 6}:
            return token == "1"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_1_output):
        key = (position, attn_1_1_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (7, "1"),
        }:
            return 0
        elif key in {(3, "0"), (3, "1"), (4, "0"), (4, "1"), (5, "1"), (7, "0")}:
            return 6
        return 3

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_1_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
        }:
            return 1
        elif key in {(2, "0"), (2, "1")}:
            return 3
        return 6

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_0_output):
        key = (num_attn_1_3_output, num_attn_0_0_output)
        if key in {
            (np.int64(1), np.int64(1)),
            (np.int64(2), np.int64(0)),
            (np.int64(2), np.int64(1)),
            (np.int64(3), np.int64(0)),
            (np.int64(3), np.int64(1)),
            (np.int64(3), np.int64(2)),
            (np.int64(4), np.int64(0)),
            (np.int64(4), np.int64(1)),
            (np.int64(4), np.int64(2)),
            (np.int64(5), np.int64(0)),
            (np.int64(5), np.int64(1)),
            (np.int64(5), np.int64(2)),
            (np.int64(5), np.int64(3)),
            (np.int64(6), np.int64(0)),
            (np.int64(6), np.int64(1)),
            (np.int64(6), np.int64(2)),
            (np.int64(6), np.int64(3)),
            (np.int64(6), np.int64(4)),
            (np.int64(7), np.int64(0)),
            (np.int64(7), np.int64(1)),
            (np.int64(7), np.int64(2)),
            (np.int64(7), np.int64(3)),
            (np.int64(7), np.int64(4)),
            (np.int64(8), np.int64(0)),
            (np.int64(8), np.int64(1)),
            (np.int64(8), np.int64(2)),
            (np.int64(8), np.int64(3)),
            (np.int64(8), np.int64(4)),
            (np.int64(8), np.int64(5)),
            (np.int64(9), np.int64(0)),
            (np.int64(9), np.int64(1)),
            (np.int64(9), np.int64(2)),
            (np.int64(9), np.int64(3)),
            (np.int64(9), np.int64(4)),
            (np.int64(9), np.int64(5)),
            (np.int64(10), np.int64(0)),
            (np.int64(10), np.int64(1)),
            (np.int64(10), np.int64(2)),
            (np.int64(10), np.int64(3)),
            (np.int64(10), np.int64(4)),
            (np.int64(10), np.int64(5)),
            (np.int64(10), np.int64(6)),
            (np.int64(11), np.int64(0)),
            (np.int64(11), np.int64(1)),
            (np.int64(11), np.int64(2)),
            (np.int64(11), np.int64(3)),
            (np.int64(11), np.int64(4)),
            (np.int64(11), np.int64(5)),
            (np.int64(11), np.int64(6)),
            (np.int64(12), np.int64(0)),
            (np.int64(12), np.int64(1)),
            (np.int64(12), np.int64(2)),
            (np.int64(12), np.int64(3)),
            (np.int64(12), np.int64(4)),
            (np.int64(12), np.int64(5)),
            (np.int64(12), np.int64(6)),
            (np.int64(12), np.int64(7)),
            (np.int64(13), np.int64(0)),
            (np.int64(13), np.int64(1)),
            (np.int64(13), np.int64(2)),
            (np.int64(13), np.int64(3)),
            (np.int64(13), np.int64(4)),
            (np.int64(13), np.int64(5)),
            (np.int64(13), np.int64(6)),
            (np.int64(13), np.int64(7)),
            (np.int64(13), np.int64(8)),
            (np.int64(14), np.int64(0)),
            (np.int64(14), np.int64(1)),
            (np.int64(14), np.int64(2)),
            (np.int64(14), np.int64(3)),
            (np.int64(14), np.int64(4)),
            (np.int64(14), np.int64(5)),
            (np.int64(14), np.int64(6)),
            (np.int64(14), np.int64(7)),
            (np.int64(14), np.int64(8)),
            (np.int64(15), np.int64(0)),
            (np.int64(15), np.int64(1)),
            (np.int64(15), np.int64(2)),
            (np.int64(15), np.int64(3)),
            (np.int64(15), np.int64(4)),
            (np.int64(15), np.int64(5)),
            (np.int64(15), np.int64(6)),
            (np.int64(15), np.int64(7)),
            (np.int64(15), np.int64(8)),
            (np.int64(15), np.int64(9)),
        }:
            return 7
        elif key in {
            (np.int64(0), np.int64(0)),
            (np.int64(0), np.int64(1)),
            (np.int64(1), np.int64(0)),
        }:
            return 1
        return 3

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_3_output):
        key = (num_attn_1_2_output, num_attn_0_3_output)
        if key in {
            (np.int64(0), np.int64(1)),
            (np.int64(0), np.int64(2)),
            (np.int64(0), np.int64(3)),
            (np.int64(0), np.int64(4)),
            (np.int64(0), np.int64(5)),
            (np.int64(0), np.int64(6)),
            (np.int64(0), np.int64(7)),
            (np.int64(0), np.int64(8)),
            (np.int64(0), np.int64(9)),
            (np.int64(0), np.int64(10)),
            (np.int64(0), np.int64(11)),
            (np.int64(0), np.int64(12)),
            (np.int64(0), np.int64(13)),
            (np.int64(0), np.int64(14)),
            (np.int64(0), np.int64(15)),
            (np.int64(1), np.int64(3)),
            (np.int64(1), np.int64(4)),
            (np.int64(1), np.int64(5)),
            (np.int64(1), np.int64(6)),
            (np.int64(1), np.int64(7)),
            (np.int64(1), np.int64(8)),
            (np.int64(1), np.int64(9)),
            (np.int64(1), np.int64(10)),
            (np.int64(1), np.int64(11)),
            (np.int64(1), np.int64(12)),
            (np.int64(1), np.int64(13)),
            (np.int64(1), np.int64(14)),
            (np.int64(1), np.int64(15)),
            (np.int64(2), np.int64(5)),
            (np.int64(2), np.int64(6)),
            (np.int64(2), np.int64(7)),
            (np.int64(2), np.int64(8)),
            (np.int64(2), np.int64(9)),
            (np.int64(2), np.int64(10)),
            (np.int64(2), np.int64(11)),
            (np.int64(2), np.int64(12)),
            (np.int64(2), np.int64(13)),
            (np.int64(2), np.int64(14)),
            (np.int64(2), np.int64(15)),
            (np.int64(3), np.int64(8)),
            (np.int64(3), np.int64(9)),
            (np.int64(3), np.int64(10)),
            (np.int64(3), np.int64(11)),
            (np.int64(3), np.int64(12)),
            (np.int64(3), np.int64(13)),
            (np.int64(3), np.int64(14)),
            (np.int64(3), np.int64(15)),
            (np.int64(4), np.int64(10)),
            (np.int64(4), np.int64(11)),
            (np.int64(4), np.int64(12)),
            (np.int64(4), np.int64(13)),
            (np.int64(4), np.int64(14)),
            (np.int64(4), np.int64(15)),
            (np.int64(5), np.int64(13)),
            (np.int64(5), np.int64(14)),
            (np.int64(5), np.int64(15)),
            (np.int64(6), np.int64(15)),
        }:
            return 3
        elif key in {(np.int64(0), np.int64(0))}:
            return 4
        return 1

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_1_output, mlp_1_0_output):
        if mlp_0_1_output in {0, 2, 4, 6}:
            return mlp_1_0_output == 7
        elif mlp_0_1_output in {1, 5}:
            return mlp_1_0_output == 0
        elif mlp_0_1_output in {3}:
            return mlp_1_0_output == 6
        elif mlp_0_1_output in {7}:
            return mlp_1_0_output == 3

    attn_2_0_pattern = select_closest(mlp_1_0_outputs, mlp_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 2, 7}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {3, 4, 5, 6}:
            return token == "4"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 5}:
            return token == "</s>"
        elif position in {1}:
            return token == "2"
        elif position in {2}:
            return token == "1"
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {6}:
            return token == "4"
        elif position in {7}:
            return token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 3, 4, 5, 7}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 5

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_0_1_output):
        if position in {0, 2, 7}:
            return attn_0_1_output == "2"
        elif position in {1}:
            return attn_0_1_output == "4"
        elif position in {3, 4, 5}:
            return attn_0_1_output == "1"
        elif position in {6}:
            return attn_0_1_output == "<pad>"

    num_attn_2_0_pattern = select(attn_0_1_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, token):
        if attn_1_1_output in {"4", "0"}:
            return token == "<s>"
        elif attn_1_1_output in {"1"}:
            return token == "1"
        elif attn_1_1_output in {"2", "3", "<s>"}:
            return token == "2"
        elif attn_1_1_output in {"</s>"}:
            return token == "0"

    num_attn_2_1_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 4, 5, 6, 7}:
            return token == "2"
        elif mlp_0_1_output in {1}:
            return token == "<pad>"
        elif mlp_0_1_output in {3}:
            return token == "</s>"

    num_attn_2_2_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {0, 2, 4, 7}:
            return attn_0_0_output == "4"
        elif mlp_0_1_output in {1}:
            return attn_0_0_output == "</s>"
        elif mlp_0_1_output in {3, 5, 6}:
            return attn_0_0_output == "3"

    num_attn_2_3_pattern = select(attn_0_0_outputs, mlp_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 7),
            ("1", 1),
            ("1", 2),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("3", 2),
            ("4", 1),
            ("4", 2),
            ("</s>", 1),
            ("</s>", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 6
        elif key in {
            ("2", 0),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("</s>", 0),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
            ("<s>", 0),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 3
        elif key in {("0", 3), ("0", 4)}:
            return 4
        return 5

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 0),
            ("0", 3),
            ("0", 4),
            ("0", 7),
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("2", 2),
            ("2", 3),
            ("3", 0),
            ("3", 2),
            ("3", 3),
            ("4", 2),
            ("4", 3),
            ("</s>", 2),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 3
        elif key in {
            ("0", 1),
            ("0", 2),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 4
        return 6

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_2_output):
        key = (num_attn_2_1_output, num_attn_2_2_output)
        if key in {
            (np.int64(0), np.int64(0)),
            (np.int64(1), np.int64(0)),
            (np.int64(2), np.int64(0)),
            (np.int64(3), np.int64(0)),
            (np.int64(4), np.int64(0)),
            (np.int64(5), np.int64(0)),
            (np.int64(6), np.int64(0)),
            (np.int64(7), np.int64(0)),
            (np.int64(8), np.int64(0)),
            (np.int64(9), np.int64(0)),
            (np.int64(10), np.int64(0)),
            (np.int64(11), np.int64(0)),
            (np.int64(12), np.int64(0)),
            (np.int64(13), np.int64(0)),
            (np.int64(14), np.int64(0)),
            (np.int64(15), np.int64(0)),
            (np.int64(16), np.int64(0)),
            (np.int64(17), np.int64(0)),
            (np.int64(18), np.int64(0)),
            (np.int64(19), np.int64(0)),
            (np.int64(20), np.int64(0)),
            (np.int64(21), np.int64(0)),
            (np.int64(22), np.int64(0)),
            (np.int64(23), np.int64(0)),
        }:
            return 5
        elif key in {
            (np.int64(0), np.int64(1)),
            (np.int64(1), np.int64(1)),
            (np.int64(2), np.int64(1)),
            (np.int64(3), np.int64(1)),
            (np.int64(4), np.int64(1)),
            (np.int64(5), np.int64(1)),
            (np.int64(6), np.int64(1)),
            (np.int64(7), np.int64(1)),
            (np.int64(8), np.int64(1)),
            (np.int64(9), np.int64(1)),
            (np.int64(10), np.int64(1)),
            (np.int64(11), np.int64(1)),
            (np.int64(12), np.int64(1)),
            (np.int64(13), np.int64(1)),
            (np.int64(14), np.int64(1)),
            (np.int64(15), np.int64(1)),
            (np.int64(16), np.int64(1)),
            (np.int64(17), np.int64(1)),
            (np.int64(18), np.int64(1)),
            (np.int64(19), np.int64(1)),
            (np.int64(20), np.int64(1)),
            (np.int64(21), np.int64(1)),
            (np.int64(22), np.int64(1)),
            (np.int64(23), np.int64(1)),
        }:
            return 0
        return 4

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_3_output, num_attn_2_1_output):
        key = (num_attn_0_3_output, num_attn_2_1_output)
        if key in {(np.int64(0), np.int64(0))}:
            return 1
        return 0

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    return logits
