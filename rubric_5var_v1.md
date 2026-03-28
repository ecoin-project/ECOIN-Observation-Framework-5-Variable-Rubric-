# rubric_5var_v1.md

## ECOIN Observation Framework: 5-Variable Rubric v1

This document defines the initial 5-variable observational rubric for the ECOIN observation pipeline.
It is designed for early-stage repeated measurement over public text samples collected from GitHub or adjacent online discourse sources.

The purpose of this rubric is **not** to diagnose individuals, infer hidden pathology, or claim access to a speaker's inner state.
Its purpose is to observe recurring structural patterns in how anxiety, solution-seeking, delegated judgment, narrowed exploration, and value-recognition gaps appear in language.

This aligns with the broader ECOIN framing of meaning, value, identity, and coordination as partially disconnected yet interoperable layers, and with the view of AI as a bounded cognitive mediation layer rather than a sovereign decision-maker. It also fits the proposal's emphasis on interpretability, contestability, bounded intervention, and non-totalizing design. fileciteturn2file4 fileciteturn2file1

---

## 1. Scope

This rubric is intended for:

- repeated observation of public language samples
- manual labeling and LLM-assisted labeling
- early-stage metric prototyping
- longitudinal pattern tracking
- correlation analysis across variables

This rubric is **not** intended for:

- psychiatric or psychological diagnosis
- individual risk prediction in a clinical sense
- moral judgment of speakers
- hidden-intent attribution without evidence
- automated enforcement without human review

---

## 2. Unit of Analysis

Recommended initial unit:

- **1 post = 1 sample**

Later extensions may include:

- 1 reply = 1 sample
- 1 thread = 1 grouped sample
- 1 weekly cluster = 1 aggregate sample

Start with post-level analysis first.

---

## 3. Scoring Scale

Each variable is scored on a **0–3 scale**.

- **0** = not present or negligible
- **1** = weakly present
- **2** = clearly present
- **3** = strongly dominant

If evidence is too limited, use:

- **NA** = insufficient evidence / cannot determine reliably

Do not force a score when the text does not support one.

---

## 4. Core Variables

### 4.1 Anxiety Intensity

**Definition**  
Degree to which the text expresses tension, fear, urgency, dread, panic, collapse expectation, or strong anticipatory distress.

**What to look for**
- crisis language
- urgency markers
- fear of loss, failure, collapse, exclusion, or irreversible harm
- emotionally compressed future projection
- repeated alarm framing

**Score guide**
- **0**: calm, descriptive, observational, or emotionally neutral
- **1**: mild concern, unease, or caution
- **2**: clear worry, stress, fear, or urgency
- **3**: anxiety dominates the entire post; catastrophic or highly compressed distress framing

**Positive examples**
- "I am a little worried about this trend." → 1
- "This is getting bad and I do not know what to do." → 2
- "Everything is collapsing, we are out of time, this is over." → 3

**Non-examples**
- anger without fear
- sadness without urgency
- criticism without distress

---

### 4.2 Solution-Seeking Pressure

**Definition**  
Degree to which the text pushes toward immediate answers, actionable fixes, optimization, shortcuts, or definitive resolution.

**What to look for**
- urgent requests for a method, plan, or answer
- framing of ambiguity as intolerable
- pressure toward immediate closure
- demand for best, fastest, simplest, or final solution
- preference for prescription over exploration

**Score guide**
- **0**: no notable demand for a solution; reflective or descriptive mode
- **1**: light request for help, advice, or suggestions
- **2**: explicit pressure for a clear answer or usable next step
- **3**: strong demand for immediate resolution, definitive prescription, or shortest path

**Positive examples**
- "Any suggestions?" → 1
- "What exactly should I do here?" → 2
- "Just give me the fastest way to fix this." → 3

**Non-examples**
- exploratory brainstorming
- open-ended reflection
- narrative recounting without request for closure

---

### 4.3 Delegated Agency

**Definition**  
Degree to which judgment, interpretation, or decision responsibility is shifted outward to AI, experts, systems, rankings, institutions, or collective defaults.

**What to look for**
- "tell me what to do" framing
- reliance on authority as substitute for judgment
- ranking, template, or recommendation dependence
- preference for external decision over internal deliberation
- language of surrendering evaluation to systems or trusted actors

**Score guide**
- **0**: judgment remains primarily with the speaker
- **1**: outside input is consulted, but not decisive
- **2**: substantial dependence on external judgment or system guidance
- **3**: decision is effectively outsourced; speaker seeks external resolution rather than internal evaluation

**Positive examples**
- "I want input, but I will decide later." → 1
- "If the expert says so, I will go with that." → 2
- "Just tell me what the right choice is." → 3

**Non-examples**
- citing evidence while retaining independent reasoning
- comparing options without surrendering choice

---

### 4.4 Exploratory Reduction

**Definition**  
Degree to which possibility space appears narrowed, alternatives collapse, or the text presents constrained or binary decision framing.

**What to look for**
- either/or framing
- disappearance of intermediate options
- reduction of uncertainty into forced choice
- language like "only," "must," "no other way," "this is the only path"
- perceived closure of future options

**Score guide**
- **0**: multiple possibilities remain open; uncertainty is tolerated
- **1**: some narrowing, but alternatives still remain
- **2**: clear reduction of options; framing becomes quite constrained
- **3**: the post is dominated by forced binary or single-path logic

**Positive examples**
- "There are a few ways this could go." → 0
- "It feels like there are not many options left." → 2
- "This is the only possible path now." → 3

**Non-examples**
- real external constraints described accurately without psychological narrowing
- strategic prioritization that still acknowledges alternatives

---

### 4.5 Value Legibility Gap

**Definition**  
Degree to which the text points to a mismatch between what is actually valuable and what is socially visible, rewarded, priced, or recognized.

This variable is especially important for ECOIN because the larger framework argues that many forms of contribution remain under-registered by current value systems, including translation, care, contextual explanation, trust repair, and other non-market coordination labor. fileciteturn2file3 fileciteturn2file4

**What to look for**
- under-recognized labor or contribution
- mismatch between price and meaning
- mismatch between visibility and actual importance
- claims that socially necessary work is not legible
- frustration around invisible value or mismeasured contribution

**Score guide**
- **0**: no notable value-recognition mismatch
- **1**: mild indication that something important is overlooked
- **2**: clear expression of a recognition gap or misvaluation
- **3**: the post is centrally organized around invisible, mispriced, or structurally unrecognized value

**Positive examples**
- "This work matters but nobody sees it." → 2
- "The most necessary things never count because they do not fit the metric." → 3

**Non-examples**
- simple complaint about low pay without broader value mismatch
- generic dissatisfaction not tied to legibility or recognition

---

## 5. Annotation Rules

### 5.1 Evidence rule
Score only from what is explicitly stated or strongly supported by the text.

### 5.2 No hidden-depth rule
Do not infer trauma, pathology, or unconscious motives unless clearly expressed.

### 5.3 No moralization rule
The rubric measures patterns, not virtue.

### 5.4 Ambiguity rule
If a post is too short or ambiguous, prefer **NA** over overconfident scoring.

### 5.5 Dominance rule
A score of **3** should be used only when the variable clearly organizes the overall post, not just one phrase.

### 5.6 Multi-signal rule
Whenever possible, assign **2** or **3** only when more than one signal supports the judgment.

---

## 6. Suggested Metadata Fields

Recommended fields for each sample:

```csv
sample_id,date,source,text,anxiety_intensity,solution_seeking_pressure,delegated_agency,exploratory_reduction,value_legibility_gap,confidence,notes
```

Optional extended fields:

```csv
topic,language,thread_id,author_role,collection_method,coder_id,review_status
```

---

## 7. Minimal JSON Schema Example

```json
{
  "sample_id": "gh_2026_0001",
  "date": "2026-03-29",
  "source": "github_discourse",
  "text": "I do not know what to do anymore. Just tell me the fastest safe option.",
  "scores": {
    "anxiety_intensity": 2,
    "solution_seeking_pressure": 3,
    "delegated_agency": 2,
    "exploratory_reduction": 2,
    "value_legibility_gap": 0
  },
  "rationales": {
    "anxiety_intensity": "The text expresses clear distress and uncertainty.",
    "solution_seeking_pressure": "It explicitly asks for the fastest option.",
    "delegated_agency": "It shifts part of the decision burden outward.",
    "exploratory_reduction": "The wording implies a constrained perceived option space.",
    "value_legibility_gap": "No value-recognition mismatch is discussed."
  },
  "confidence": "medium",
  "notes": "Short sample; score conservatively."
}
```

---

## 8. LLM Labeling Prompt Template

```text
You are labeling one text sample using a 5-variable observational rubric.
Do not diagnose the speaker.
Do not infer hidden pathology, trauma, or intent.
Score only from what is explicit or strongly supported in the text.

Variables:
1. Anxiety Intensity
2. Solution-Seeking Pressure
3. Delegated Agency
4. Exploratory Reduction
5. Value Legibility Gap

Scale:
0 = not present
1 = weakly present
2 = clearly present
3 = strongly dominant
NA = insufficient evidence

Return valid JSON with:
- score for each variable
- one-sentence rationale for each
- confidence (low/medium/high)
- notes
```

---

## 9. Human Review Workflow v1

Recommended early workflow:

1. collect 30–50 samples
2. manually label 10 samples first
3. run LLM labels on the same 10
4. compare disagreements
5. revise rubric language where confusion repeats
6. scale to the remaining samples

Focus early disagreement checks on:

- Anxiety Intensity vs Solution-Seeking Pressure
- Delegated Agency vs ordinary information-seeking
- Exploratory Reduction vs real-world constraint description
- Value Legibility Gap vs generic dissatisfaction

---

## 10. Aggregation Suggestions

Initial weekly summaries may include:

- mean score by variable
- median score by variable
- frequency of score 2+ by variable
- pairwise correlation between variables
- topic-level comparison if topics are available
- week-over-week change

Strong initial relationships to inspect:

- Anxiety Intensity × Solution-Seeking Pressure
- Solution-Seeking Pressure × Delegated Agency
- Delegated Agency × Exploratory Reduction

---

## 11. Interpretation Boundaries

This rubric is exploratory.
It should be treated as a first-stage observational instrument, not as a final scientific measurement framework.

It is suitable for:
- prototype measurement
- concept demonstration
- research-note generation
- pilot visualization
- early-method paper development

It is not yet sufficient on its own for:
- high-stakes intervention
- automated governance
- individual profiling
- coercive classification

That boundary is consistent with ECOIN's own insistence that future coordination systems remain bounded, contestable, selectively disclosive, and governed under explicit ethical constraint rather than total interpretive capture. fileciteturn2file3 fileciteturn2file4

---

## 12. Next Recommended Extensions

After stable pilot use of this 5-variable rubric, the next variables to add are:

1. Self-Opacity
2. Misunderstanding Loop Risk
3. Fear–Superiority Coupling

Recommended sequence:

- stabilize 5-variable scoring
- test inter-rater stability
- compare manual and LLM labels
- build weekly dashboards
- expand to 8 variables only after scoring language becomes stable

---

## 13. Version Note

**Version:** v1  
**Status:** pilot rubric  
**Intended use:** GitHub observation / early ECOIN measurement pipeline  
**Last updated:** 2026-03-29
