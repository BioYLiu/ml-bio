
def compute_hmm(model, sequence):

    input_states = sequence['Z']
    input_emissions = sequence['X']
    # the first state
    i = 0
    PI = model.pi( model.index_hidden_state(input_states[i]) )
    S = PI
    # first hidden and emission nodes
    S +=  model.emission(
            model.index_hidden_state(input_states[i]),
            model.index_observable(input_emissions[i])
        )


    for i in range(1, len(input_states)):
        ### patch project 2, avoiding logzero with -infintity
        S += model.transition(
                    model.index_hidden_state(input_states[i - 1]),
                    model.index_hidden_state(input_states[i])
            )
        S += model.emission(
                model.index_hidden_state(input_states[i]),
                model.index_observable(input_emissions[i])
            )

    return S


def merge_array_of_sequences(data):
    """Returns a dictionary from an array of dictionaries"""
    return { index:value for inner_dict in data for index, value in inner_dict.items() }
