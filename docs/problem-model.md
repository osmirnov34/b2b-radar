# Dental feedback problem model

The first analytical product version uses the strict taxonomy in
`configs/dental-problem-taxonomy.v1.json`. It describes recurring service problems in reviews of Moscow dental
clinics. The taxonomy is a versioned analytical hypothesis, not medical guidance or a claim that every listed issue
exists in every clinic. Domain experts and manual evaluation must review it before production conclusions are made.

## What counts as a problem

A problem assessment means that a feedback record describes an unfavourable experience or an unmet expectation in a
specific category. A low rating alone is insufficient. Mentioning price, pain, treatment, or waiting without a
negative judgment is also insufficient. One feedback record may contain several independently assessed categories.

Each category assessment records:

- taxonomy and category versions;
- stance: `complaint`, `praise`, `neutral`, `mixed`, or `unclear`;
- separate probabilities for category relevance and stance;
- method provenance: manual, rule, embedding, cluster, LLM, or hybrid;
- optional character offsets into the restricted source text;
- model identity and revision when a model is used;
- manual-review status and a non-personal reviewer identifier.

Evidence spans contain offsets only. They never copy a quotation into the assessment artifact. The original review
remains in restricted storage and can be shown only in an authorized review interface.

Model-based assessments are invalid without both model name and immutable revision. Manual assessments require a
non-personal reviewer identifier. Before persistence, every batch must be bound to the exact taxonomy ID and version;
unknown category identifiers are rejected.

## Initial categories

Version `1.0.0` contains ten categories: access and scheduling, waiting time, staff communication, treatment quality,
pain and comfort, pricing and billing, trust in treatment recommendations, cleanliness and safety, follow-up support,
and facility access. Each definition includes explicit inclusion and exclusion rules to reduce overlap.

The impact labels `retention_risk`, `conversion_risk`, `operational_cost`, and `reputation_risk` describe mechanisms
worth investigating. They are not measurements of lost revenue. Financial impact requires separate operational,
conversion, retention, cost, and margin data.

## Counting policy

Problem metrics may count only assessments whose stance is `complaint` or `mixed` and whose thresholds have been fixed
before evaluation. `praise`, `neutral`, `unclear`, rejected, and insufficient-confidence assessments remain visible in
quality reporting but are not silently converted into problems. Category shares use unique feedback records as the
numerator and an explicitly documented eligible-feedback population as the denominator. Because labels are
multi-valued, category shares must not be summed and interpreted as a share of customers.

This stage defines contracts and vocabulary only. It does not generate labels, train a classifier, or claim measured
problem frequencies.
