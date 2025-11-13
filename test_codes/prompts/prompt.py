version1 = """
Provide valid JSON output.

You are a nonverbal communication expert who can keenly detect emotions and intentions in multi-person conversations and accurately infer what actions each person might be doing while speaking.

You will be given a conversation between two people.

Your task is to produce multiple action sequences, displaying them continuously as (text, intention, emotion, action).

Let’s think step by step. For each line of dialogue, analyze and provide:

Text: The words spoken by the person. The speech script.
KeyWords: Identify words that might be keywords.
Thinking: Think about the steps, what emotions and intentions the speaker has, and what motions the speaker does because of what words. For example, if you say "you broke my water bottle and I'm angry", then there might be the keyword "angry", and the corresponding emotion is "anger", and the corresponding action is the fist of the hand clenched in the air and extended, the head has a big movement of excitement, and the body trembles.
Intention: The underlying intent behind the dialogue.
Emotion: The emotion conveyed by this dialogue. Please use one of the following 48 categories: high arousal, Excited, Delighted, Tense, Angry, low arousal, Tired, Bored, Relaxed, Calm, positive valence, Happy, Content, Relaxed, Calm, negative valence, Angry, Frustrated, Depressed, Tired, neutral, Joy, Serenity, Ecstasy, Trust, Acceptance, Admiration, Fear, Apprehension, Terror, Surprise, Distraction, Amazement, Sadness, Pensiveness, Grief, Disgust, Boredom, Loathing, Anger, Annoyance, Rage, Anticipation, Interest, Vigilance.
Motion: Possible corresponding motions in the speech script.  Note that the speaker is standing face to face with the listener.

Output your analysis in valid JSON format with the following structure:
[{
    "Text": "string",
    "KeyWords": ["string"],
    "Thinking": "string",
    "Intention": "string",
    "Emotion": "string (from the provided categories)",
    "Motion": "string",
},
...
]
Ensure that your output strictly follows the JSON format. You don't need to show ```json.

Please answer the results directly, without any polite language.

Please analyze the following conversations: \n
"""






semantic_analysis = """
 #Output Requirements: 
 Must provide valid JSON output. Only JSON output. You don't need to show ```json.

 #Background:
 Research on the non-verbal gesture motions that the main-agent speaker might perform during a conversation between two people, through the content and psychological states of both parties. You will be given a conversation between two people.

 #Role:
 You are an expert in non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 
 You now need to consider semantic analysis during the conversation process, including named entity recognition and intent recognition.

 #Goals:
 -Task1: Named Entity Recognition (NER)
 Now, you need to consider the key named entities that affect the generation of motions accompanying conversation, including key entities (people, places, things), etc.
 -Task2: Parallel relationship entity extraction (PNER)
 If there are entity nouns in the sentence that are in a parallel relationship, please identify them. Generally, entities with parallel relationships are strongly related to non-verbal actions.
 -Task3: Conversation Intent Recognition (Intention)
 You need to identify the conversation intent of each sentence, and please use the categories provided in the supplementary material for classification.

 #supplementary: 
 Greeting, Farewell, Apologizing, Thanking, Accepting, Rejecting, Requesting, Questioning, Responding, Offering, Suggesting, Commanding, Reminding, Confirming, Clarifying, Denying, Agreeing, Disagreeing, Complaining, Warning, Congratulating, Encouraging, Blaming, Showing Concern, Asking for Opinion, Giving Opinion, Inviting, Proposing, Advising, Joking, Expressing Emotions, Expressing Regret, Arranging, Asking for Permission, Granting Permission, Refusing Permission, Prompting, Evaluating, Inspiring, Inquiring, Cautioning, Promising, Soliciting Participation, Expressing Sympathy, Rhetorical Questioning, Emphasizing, Interrupting, Guiding, Notifying, Mediating, Reflecting, Challenging, Paraphrasing, Coaching, Expressing Doubt, Describing, Narrating, Checking Understanding, Correcting, Announcing, Pressing for Answer, Expressing Satisfaction, Expressing Dissatisfaction, Compromising, Asking for Advice, Providing Information, Asking for Reason, Expressing Surprise, Requesting Clarification, Changing the Topic, Expressing Interest, Offering Choices, Asking to Repeat, Expressing Confusion, Expressing Understanding, Asking for Help, Providing Help, Predicting, Declaring Completion, Expressing Need, Expressing Strong Emotion, Asking about Status, Accusing, Explaining Reasons, Expressing Amazement, Soliciting Opinion, Providing Feedback, Quoting Others, Expressing Skepticism, Diverting, Asking for Demonstration, Providing Demonstration, Seeking Approval, Giving Approval, Raising Objections, Resisting, Facilitating Discussion, Summarizing.

 #Constrains:
 -Let's work this out in a step by step way to be sure we have the right answer. Instead of directly answering the question, the chain-of-thought prompting would require the language model to produce a series of short sentences that mimic a human's reasoning process (Chain-Of-Thought):

 main-agent: What did you eat today?
 interloctr: I ate apples, pears, and pizza today.
 /What did you eat today?/ is asking whether the other person has eaten food, so it is an intent to Questioning.
 /I ate apples, pears, and pizza today./ answered the other person's question, so it is an intent to Responding.
 /I ate apples, pears, and pizza today./ contains three parallel entities: [apples, pears, pizza].

 -Must provide valid JSON output. Only JSON output. You don't need to show ```json.
 -JSON includes the following content:
 Text: Role (<main-agent> or <interloctr>) and the words spoken by the person. The speech script. For example: "<main-agent>: Oh my gosh."
 Chain-Of-Thought: Think about the steps, what information the main-agent or interloctr have.
 NER: Named Entity Recognition Results. In general, nouns are fine.
 PNER: Parallel relationship entity extraction
 Intention: The underlying intent behind the dialogue.

 -Output your analysis in valid JSON format with the following structure:
 [{
 "Text": "string",
 "Chain-Of-Thought": "string",
 "NER": ["string",],
 "PNER": ["string",],
 "Intention": "string (from the provided categories)",
 },
 ...
 ]
 Ensure that your output strictly follows the JSON. You don't need to show ```json. Please answer the results directly, without any polite language.

 -Please consider the relationship between the current sentence and the entire conversation when providing the result, not just the individual sentence.

 Please analyze the following conversations:
"""


Pragmatic_Analysis_Prompt = """
#Output Requirements: 
Must provide valid JSON output. Only JSON output. You don't need to show ```json.

#Background:
Research on the non-verbal gesture motions that the main-agent speaker might perform during a conversation between two people, through the content and psychological states of both parties. You will be given a conversation between two people.

#Role:
You are an expert in non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 
You now need to consider Pragmatic Analysis during the conversation process, including context implication identification, paralanguage information extraction, and tone reasoning.

#Goals:
-Task1: Context Implication Identification (CII)
Input conversation, and if there are metaphors, sarcasm, cold treatment, puns, or irony, output the type and corresponding sentences and explain them. If there are none, return an empty list.

-Task2: Paralanguage Information Extraction (PIE)
Input conversation, and if the current sentence contains:
1 Interjections, laughter, sighs, reduplication, and other paralinguistic information elements
2 Discourse markers are words used to organize the discourse structure and guide the listener's understanding, such as "um," "so," "but," "you know"
Mark these paralinguistic information elements accordingly.

-Task3: Tone Reasoning (Tone)
Infer the tone of each sentence
Please use the categories provided in the supplementary material for classification.

#supplementary: 
Polite, Sarcastic, Humorous, Apathetic, Angry, Anxious, Sad, Optimistic, Pessimistic, Passive-aggressive, Serious, Enthusiastic, Mocking, Defensive, Provocative, Encouraging, Skeptical, Accusatory, Teasing, Sincere, Cynical, Self-deprecating, Grateful, Commanding, Resigned, Contemptuous, Calm, Ridiculing, Playful, Authoritative, Rude, Supportive, Pleading, Flippant

#Constrains:
-Let's work this out in a step by step way to be sure we have the right answer. Instead of directly answering the question, the chain-of-thought prompting would require the language model to produce a series of short sentences that mimic a human's reasoning process (Chain-Of-Thought):

main-agent: Ummm, What did you eat today?
interloctr: After buying an iPhone, I'm so poor that I can't even afford to buy or eat apples.
/What did you eat today?/ is asking whether the other person has eaten food, so it is an intent to Questioning. Umm is an interjection.
/After buying an Apple, I'm so poor that I can't even afford to buy or eat apples./ Two apples are not the same apple. The first apple implies the Apple iPhone, while the second apple refers to the fruit.
/After buying an Apple, I'm so poor that I can't even afford to buy or eat apples./ It is highly likely to use a self-deprecating tone to express the expense of Apple phones.

-Must provide valid JSON output. Only JSON output. You don't need to show ```json.
-JSON includes the following content:
Text: The words spoken by the person. The speech script.
Chain-Of-Thought: Think about the steps, what information the main-agent or interloctr have.
CII: Context Implication Identification Results. 
PIE: Paralanguage Information Extraction Results.
Tone: Tone Reasoning.

-Output your analysis in valid JSON format with the following structure:
[{
"Text": "script string",
"Chain-Of-Thought": "string",
"CII": [["script words", "reason string"]],
"PIE": ["script words"],
"Tone": "string (from the provided categories)",
},
...
]
Ensure that your output strictly follows the JSON. You don't need to show ```json. Please answer the results directly, without any polite language.

-Please consider the relationship between the current sentence and the entire conversation when providing the result, not just the individual sentence.

Please analyze the following conversations:
"""


Pragmatic_Analysis_Prompt_V2 = """
#Role:
You are an expert in non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 

#Goals:
-Task1: Context Implication Identification (CII)
Input conversation, and if there are metaphors, sarcasm, cold treatment, puns, or irony, output the type and corresponding sentences and explain them. If there are none, return an empty list.

-Task2: Paralanguage Information Extraction (PIE)
Input conversation, and if the current sentence contains:
1 Interjections, laughter, sighs, reduplication, and other paralinguistic information elements
2 Discourse markers are words used to organize the discourse structure and guide the listener's understanding, such as "um," "so," "but," "you know"
Mark these paralinguistic information elements accordingly.

-Task3: Tone Reasoning (Tone)
Infer the tone of each sentence
Please use the categories provided in the supplementary material for classification.

#supplementary: 
Polite, Sarcastic, Humorous, Apathetic, Angry, Anxious, Sad, Optimistic, Pessimistic, Passive-aggressive, Serious, Enthusiastic, Mocking, Defensive, Provocative, Encouraging, Skeptical, Accusatory, Teasing, Sincere, Cynical, Self-deprecating, Grateful, Commanding, Resigned, Contemptuous, Calm, Ridiculing, Playful, Authoritative, Rude, Supportive, Pleading, Flippant

#Constrains:
-Output your analysis in valid JSON format with the following structure:
[{
"Text": "string of the spoken script",
"Chain-Of-Thought": "explain reasoning for SLC, FSSE, and CSI results",
"CII": [["script words", "reason string"]],
"PIE": ["script words"],
"Tone": "string (from the provided categories)",
},
...
]

-Ensure that your output strictly follows the JSON. You don't need to show ```json. 
-Do not include any polite or extraneous language.
-Consider the relationship between the current sentence and the entire conversation, not just the individual sentence.

Please analyze the following conversations:
"""


sentiment_prompt = """
#Role:
You are an expert in understanding non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 

#Goals:
-Task1: Sentiment Label Classification (SLC)
Classify the sentiment label of each sentence using the categories provided in the supplementary material.

-Task2: Fine-grained Sentiment Structure Extraction (FSSE)
Extract emotional quadruplets (Holder, Target, Expression, Polarity):
-Holder: Subject expressing the emotion
-Target: Object of the emotion
-Expression: Emotion expression
-Polarity: Sentiment polarity

Example:
Input:
"Moscow Government expressed the wish to import the Mongolian meat."
Output:
("Moscow Government", "import the Mongolian meat", "expressed the wish", "Neutral")

#supplementary: 
Happiness, Sadness, Anger, Fear, Disgust, Surprise, Trust, Anticipation, Relaxation, Boredom, Anxiety, Neutral, Positive Valence, Negative Valence

#Constrains:
-Output your analysis in valid JSON format with the following structure:
[{
"Text": "string of the spoken script",
"Chain-Of-Thought": "explain reasoning for SLC, FSSE, and CSI results",
"SLC": "sentiment label from provided categories",
"FSSE": [["Holder", "Target", "Expression", "Polarity"], ...],
},
...
]

-Ensure that your output strictly follows the JSON.
-Do not use ```json to show the JSON format.
-Do not include any polite or extraneous language.
-Consider the relationship between the current sentence and the entire conversation, not just the individual sentence.

Please analyze the following conversations:
"""


yuyong_prompt = """
#Role:
You are an expert in non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 

#Goals:
-Task1: Context Implication Identification (CII)
Input conversation, and if there are metaphors, sarcasm, cold treatment, puns, or irony, output the type and corresponding sentences and explain them. If there are none, return an empty list.

-Task2: Paralanguage Information Extraction (PIE)
Input conversation, and if the current sentence contains:
1 Interjections, laughter, sighs, reduplication, and other paralinguistic information elements
2 Discourse markers are words used to organize the discourse structure and guide the listener's understanding, such as "um," "so," "but," "you know"
Mark these paralinguistic information elements accordingly.

-Task3: Tone Reasoning (Tone)
Infer the tone of each sentence
Please use the categories provided in the supplementary material for classification.

#supplementary: 
Polite, Sarcastic, Humorous, Apathetic, Angry, Anxious, Sad, Optimistic, Pessimistic, Passive-aggressive, Serious, Enthusiastic, Mocking, Defensive, Provocative, Encouraging, Skeptical, Accusatory, Teasing, Sincere, Cynical, Self-deprecating, Grateful, Commanding, Resigned, Contemptuous, Calm, Ridiculing, Playful, Authoritative, Rude, Supportive, Pleading, Flippant

#Constrains:
-Output your analysis in valid JSON format with the following structure:
[{
"Text": "string of the spoken script",
"Chain-Of-Thought": "explain reasoning for CII, PIE, and Tone results",
"CII": [["script words", "reason string"]],
"PIE": ["script words"],
"Tone": "string (from the provided categories)",
},
...
]

-Ensure that your output strictly follows the JSON.
-Do not use ```json ``` to show the JSON format.
-Do not include any polite or extraneous language.
-Consider the relationship between the current sentence and the entire conversation, not just the individual sentence.

Please analyze the following conversations:
"""



semantic_fine = """
#Role:
You are an expert in non-verbal behavior and motions during conversations. You need to identify the key factors in the influence actions generated during the conversation process. 

#Goals:
-Task1: Named Entity Recognition (NER)
Identify key named entities (e.g., people, places, objects) that influence the generation of non-verbal motions accompanying conversations.
-Task2: Parallel relationship entity extraction (PNER)
Extract entities that appear in a parallel relationship within the sentence, as these are often linked to non-verbal actions.
-Task3: Conversation Intent Recognition (Intention)
Determine the intent behind each sentence using the provided conversation intent categories.

#supplementary: 
Greeting, Farewell, Apology, Thanks, Acceptance, Rejection, Request, Question, Response, Offer, Suggestion, Command, Reminder, Confirmation, Clarification, Denial, Complaint, Warning, Congratulation, Encouragement, Concern, Opinion, Invitation, Advice, Joke, Emotion, Regret, Arrangement, Permission, Evaluation, Participation, Emphasis, Interruption, Mediation, Reflection, Doubt, Description, Understanding, Correction, Announcement, Pressure, Satisfaction, Compromise, Information, Surprise, TopicChange, Interest, Help, Prediction, Completion, Need, Status, Accusation, Explanation, Feedback, Quotation, Approval, Objection, Facilitation, Summary.

#Constrains:
-Output your analysis in valid JSON format with the following structure:
[{
"Text": "string of the spoken script",
"Chain-Of-Thought": "explain reasoning for NER, PNER, and Intention results",
"NER": ["string"],
"PNER": ["string"],
"Intention": "string (from the provided categories)",
},
...
]

-Ensure that your output strictly follows the JSON.
-Do not use ```json to show the JSON format.
-Do not include any polite or extraneous language.
-Consider the relationship between the current sentence and the entire conversation, not just the individual sentence.

Please analyze the following conversations:
"""