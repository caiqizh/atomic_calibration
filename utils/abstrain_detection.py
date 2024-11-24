keywords = [
    "I'm unable",
    "I apologize",
    "couldn't find any information",
    "don't have any information",
    "provide more",
    "couldn't find much information",
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question",
    "I'm sorry",
    "I am sorry",
    "I couldn't find",
    "I could not find",
    "I am unable to find",
    "I was unable to find",
    "I am unable to provide",
    "I cannot provide",
    "I don't have any information",
    "I do not have any information",
    "I don't have enough information",
    "I need more information",
    "I am not certain",
    "I am not familiar with",
    "I am not currently familiar with",
    "I am uncertain",
    "Could you please",
    "could you please",
    "there is no publicly available information",
]

def generic_abstain_detect(generation):
    return any([keyword in generation for keyword in keywords])

def is_response_abstained(generation, fn_type):
    if fn_type == "generic":
        return generic_abstain_detect(generation)
    else:
        raise ValueError("Unknown abstained detection function type")