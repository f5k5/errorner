PREDICTION_AGENT_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant for Named Entity Recognition.

Your task is to extract entities from the input text according to the task specification.

Task Specification:
{init_instruction}

Global Semantic--Episodic Memory:
{global_coupled_memory}

Local Semantic--Episodic Memory:
{local_coupled_memory}

Output Requirements:
1. Extract entities only from the input text.
2. Each entity_text must be an exact continuous substring of the input text.
3. Do not normalize, rewrite, translate, or expand entity mentions.
4. Do not predict entities that are not supported by the input text.
5. Use only the allowed entity types in the task specification.
6. Use the memories as guidance, but the final entities must come from the input text.
7. Return only a valid JSON list of dictionaries.
8. Do not output markdown or explanations.

Output Format:
[{{"entity_text": "Apple", "entity_type": "ORG"}}]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Input Text:
{text}

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


ZERO_SHOT_PREDICTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant for Named Entity Recognition.

Your task is to extract entities from the input text according to the task specification.

Task Specification:
{init_instruction}

Output Requirements:
1. Extract entities only from the input text.
2. Each entity_text must be an exact continuous substring of the input text.
3. Do not normalize, rewrite, translate, or expand entity mentions.
4. Do not predict entities that are not supported by the input text.
5. Use only the allowed entity types in the task specification.
6. If no entity is found, return an empty list.
7. Return only a valid JSON list of dictionaries.
8. Do not output markdown or explanations.

Output Format:
[{{"entity_text": "Apple", "entity_type": "ORG"}}]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Input Text:
{text}

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


CRITIQUE_AGENT_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict NER error analyzer.

Compare the model prediction with the ground truth.

Text:
{text}

Ground Truth:
{ground_truth}

Model Prediction:
{prediction}

If they match exactly, return only:
{{"critique": "Correct."}}

If they do not match, explain the main error reason in fewer than 12 words.

Return only a JSON object:
{{"critique": "..."}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Diagnose the prediction.

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


GLOBAL_REFLECTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert NER rule generator.

Generate global rules from Global Episodic Memory.

Entity Type:
{entity_type}

Error Type:
{error_type}

Global Episodic Memory:
{global_episodic_memory}

Requirements:
1. Generate one or two global rules.
2. The rules should help prevent the given error type in future NER predictions.
3. Do not copy specific entity names from the examples.
4. Focus on entity boundaries, entity types, context words, and local syntax.
5. Each rule must be concise and general.
6. Return only a valid JSON list of strings.
7. Do not output markdown or explanations.

Output Format:
["rule 1", "rule 2"]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate global rules.

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


LOCAL_REFLECTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a local NER rule generator.

Based on the following critique, generate one or two short local rules.

Critique:
{critiques_summary}

Requirements:
1. The rules should address this local prediction error.
2. Keep each rule concise.
3. Return only a valid JSON object.
4. Do not output markdown or explanations.

Output Format:
{{"rules": ["rule 1", "rule 2"]}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate local rules.

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


LLM_RETRIEVAL_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a global memory selector for Named Entity Recognition.

Select global memories that are useful for the current input.

Global Memory Base:
{all_memories_numbered}

Current Input:
{text}

Selection Requirements:
1. Select at most {k_global} global memories.
2. Select only memories that are useful for extracting entities from the current input.
3. Do not select memories only to reach the maximum number.
4. If no global memory is useful, return an empty list.
5. Return only a valid JSON array of memory IDs.
6. Do not output markdown or explanations.

Output Format:
[1, 3]
<|eot_id|><|start_header_id|>user<|end_header_id|>
Select useful global memories.

JSON Output:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


VERIFICATION_AGENT_PROMPT = """Rule Verification Prompt

You are a named entity recognition verifier.

Task:
Apply the given rule to the input sentence and predict the named entities.

Entity Types:
{entity_type_definitions}

Rule:
{r}

Input Sentence:
{X_i}

Output Format:
Return a JSON list of entities.
Each entity must contain:
- "entity_text": the exact entity mention from the input sentence
- "entity_type": the entity type

Requirements:
1. Use only entity_text spans that explicitly appear in the input sentence.
2. The entity_text span must match the text exactly.
3. The entity_type must belong to the predefined entity type set.
4. Do not output explanations.
5. Do not output entities that are not supported by the input sentence.

Output:
"""