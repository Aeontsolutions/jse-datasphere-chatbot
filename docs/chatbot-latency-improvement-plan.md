> 📝 **Revision history**
>
> - **2026-10-22 (afternoon):** First version. We wrote it after the benchmark research. We first described the chatbot's steps wrongly, so this version uses the steps in the code.

> 🎯 **Purpose:** Show how to move the JSE DataSphere chatbot closer to the industry standard for response time, in order of effort. Audience: product, leadership and engineering.
>
> 📊 **Background:** `docs/chatbot-latency-industry-benchmarks.md` (benchmark research) and the production log table `jse_raw_financial_data_dev_elroy.interactions` (our own numbers).

---

## Summary

Some users say the chatbot is slow. The numbers partly agree. The typical answer takes 10.4 seconds, which is close to what users accept in published studies. The slowest answers take much longer: 1 in 10 takes more than 26.8 seconds, and 1 in 20 takes more than 32.3 seconds. Users also see nothing on screen until the full answer is ready.

There is no formal industry standard for chatbot response time. This plan uses the best evidence we found: user-research limits, one controlled study, and the response times of similar products. The plan does three things, in this order. It shows users progress within about 1 second. It finds out why the slowest answers are slow. It then makes answers faster only where we can show that answer quality stays the same.

> ⚠️ **The key number:** 1 in 20 answers takes more than 32.3 seconds (188 requests from real users, 15 August to 16 October 2026). The most recent month is worse, but it has only 26 requests, so we cannot call it a trend yet.

---

## The situation today

**Where it works well**
- The typical answer (median) takes 10.4 seconds. A 2026 study of 240 people tested waits of 2, 9 and 20 seconds. A press summary of the study says answers after the 9-second wait were rated most useful. We have not checked this detail in the paper itself.
- Intercom's Fin support agent reports a typical time to first word of about 7 seconds (reported by Intercom). Our time is for the full answer, so the two are not equal. The ranges are similar.
- Finance research agents that search the web took 91 to 109 seconds on average for public-company questions. We are much faster than they are.

**Where it falls short**
- The slowest 10% of answers take more than 26.8 seconds. The slowest 5% take more than 32.3 seconds. A usability rule from Nielsen Norman Group says that a wait over 10 seconds needs visible progress. We show none.
- The chatbot sends one reply after all the work is done. During the wait the user sees no sign of progress.
- Search-based AI answers that users know well are faster. The Perplexity Sonar service has a typical time of 3.2 seconds, and 95 of every 100 answers take 9.1 seconds or less (measured by a third party). We are about 3 times slower.
- In October, 1 in 20 answers took more than 58.3 seconds (26 requests). Our load balancer ends a request after 60 seconds of silence. The slowest answers are close to that limit.

**What the chatbot does today**
Each question goes through three steps. A fast AI call decides if the question is in scope. At the same time, a second AI call finds company ticker symbols. Then one main AI call writes the answer, with Google Search if web search is on. We have not yet measured how long each step takes. That is the first thing this plan fixes.

**The bottom line**
The average answer is acceptable. The slow answers and the empty screen are the real problems. Both can be improved before we change any AI model.

---

## The plan in one picture

| Stage | What | Effort | Expected impact |
|---|---|---|---|
| **1. Quick wins** | Measure each step. Check which AI model is really answering. Check why no answer comes from the cache. | 1 to 2 weeks | We learn where the time goes. No change for users yet. |
| **2. Best practices** | Show progress messages within about 1 second. Test a lower thinking setting and tighter answers. | 2 to 4 weeks | Users see progress at once. The typical answer may be several seconds faster (rough estimate, not tested). |
| **3. Experimental upgrades** | Try a faster AI model. Test web search only when needed. | 4 to 6 weeks | Possible gain in the slowest answers. Needs quality checks first. |
| **4. Long-term bets** | Show the answer word by word as it is written. | Later | Largest gain in how fast the chatbot feels. Needs front-end work. |

Each stage starts only when the one before it is measured. The gates are in the sections below.

---

## Stage 1: Quick wins (1 to 2 weeks, low risk)

These changes cost little and tell us where to spend effort in the later stages. We should not choose Stage 2 or Stage 3 work before we have this data.

### Three concrete changes

**1. Measure the time of each step**
- **Today:** We record one total time per request. We cannot say which step makes a slow answer slow.
- **Change:** The code now records the time of the routing step, the symbol step and the answer step for every `/chat/stream` request. It is committed on this branch and is not deployed yet. Deploy it, then collect at least 2 weeks of real requests.
- **Why it helps:** If the answer step (which includes Google Search) is the slow one, the fix is different from the fix for a slow routing step. We stop guessing.

**2. Check which AI model answers**
- **Today:** The code asks for `gemini-3.7-flash`. Google's documentation (updated 13 October 2026) says this name is sent to `gemini-3.8-flash`. We have not confirmed that this happens for our traffic.
- **Change:** Find the model name in the real replies from Google, or in the Google Cloud usage data. If it is 3.8, decide whether to keep it or to choose a model on purpose.
- **Why it helps:** Independent tests (Artificial Analysis, a benchmark site) show 3.8 Flash is slower and writes longer answers. This could be one cause of the worse October numbers.

**3. Check why the answer cache is not used**
- **Today:** 0 of 1,228 production requests to `/chat/stream` came from the cache. Among the 188 real-user requests, 27 repeated an earlier question (161 different questions, ignoring capital letters and extra spaces). So the possible gain is small, but it is not zero.
- **Change:** Find out whether the cache is on in production and whether it works. Do not change it yet.
- **Why it helps:** A cached answer would be far faster than a new one. If the cache is off by mistake, this is the cheapest speed gain available for repeated questions. If it is off on purpose (for example because financial data changes), we write down why.

### Expected outcome of Stage 1
No change in speed. We get a clear list of which step costs the most time. Risk: low. One caution: 2 weeks of data may still be too little for a reliable 95th percentile. We should wait until we have at least 200 requests.

> **Gate to Stage 2:** We have step times for at least 200 real requests and we know which model answers.

---

## Stage 2: Best practices (2 to 4 weeks, medium risk)

### Two concrete changes

**1. Show progress messages within about 1 second**
- **Today:** The chatbot sends one reply when all work is done. The user waits 10 to 58 seconds with no sign of progress.
- **Change:** Send short status messages while the work is in progress, for example "Checking your question", "Looking up JSE information", "Writing the answer". The first message goes out within about 1 second. Each message also keeps the connection open, so the 60-second limit of the load balancer no longer ends long requests.
- **Why it helps:** Usability research says that waits over 10 seconds need visible progress. This does not make the answer faster. It makes the wait feel shorter and it removes the 60-second error risk. The web or app team must also show the messages (see "What We Need").

**2. Test a lower thinking setting and tighter answers**
- **Today:** The answer step uses the `LOW` thinking level and can write up to 8,192 tokens. On average a request produces 600 to 1,000 thinking tokens and 500 to 800 answer tokens (a token is a small piece of text, about three-quarters of a word).
- **Change:** Test a lower thinking level and a clear length limit on the answer. Run the existing evaluation suite (our set of test conversations that score answer quality) to compare quality before and after.
- **Why it helps:** Waiting time grows roughly in line with the number of tokens the AI writes. OpenAI's guide says that cutting output by half can cut waiting time by about half. No published number exists for Gemini's `LOW` level, so we must measure it.

### Expected outcome of Stage 2
Users see progress at once. The typical answer may become faster. We do not yet have a tested estimate of how much. Risk: shorter answers or less thinking could lower the quality of financial answers. The evaluation suite is the gate. Nothing ships if quality falls.

> **Gate to Stage 3:** Progress messages are live. The test of thinking level and answer length is measured against the evaluation suite.

---

## Stage 3: Experimental upgrades (4 to 6 weeks, higher uncertainty)

### Two concrete changes

**1. Try a faster AI model for the answer**
- **Today:** One model writes every answer.
- **Change:** Test one or two faster models, for example a Flash-Lite model, with the evaluation suite. Independent tests show a Flash-Lite model writing about 433 to 447 tokens per second, against about 137 to 208 for Flash models. These tests use a different task and a different thinking setting from ours.
- **Why it helps:** Faster writing shortens the main step. Quality for financial answers is not yet known, so no model is chosen before the quality check passes. AI models are also retired every 1 to 2 months, so any choice needs a retirement-date check.

**2. Use web search only when a question needs it**
- **Today:** Web search is on by default. Google Search runs inside the answer step.
- **Change:** If the Stage 1 data shows that search makes up most of the answer time, test a rule that turns search off for questions that do not need current information.
- **Why it helps:** Search adds time to every answer that uses it. We do not know how much time. The Stage 1 data will tell us. If the number is small, we skip this change.

### Expected outcome of Stage 3
A possible gain in the slowest answers. We cannot give a range today, because no source measures these changes on our setup.

> **Gate to Stage 4:** Stage 3 results are measured, and the progress messages have shown that the front end can show live information.

---

## Stage 4: Long-term bets (later, highest potential)

**Show the answer word by word**
- **Today:** The user gets the whole answer at once.
- **Change:** Send the answer text as the AI writes it, as ChatGPT does.
- **Why it helps:** The user starts reading after the first second or two. The total time does not change, but the wait feels much shorter. The thinking time before the first word remains. Streaming also needs front-end work and care with the structured parts of the reply, such as sources and charts.

---

## Important caveat: the evidence is thinner than it looks

- **Small samples.** Our overall figures come from 188 requests, and the 95th percentile is the time that 95 of every 100 answers beat. The October figures come from 26 requests, so one or two slow answers can change them a lot. We should not read October as a trend until we have more data.
- **No formal standard.** We found no published response-time target for chatbots from the large vendors. This does not prove that none exists. The comparison uses research studies, third-party measurements and user reports. Several of these were read through summaries, so the exact numbers should be checked before we quote them outside the company.
- **Benchmarks come from other products.** Perplexity and Intercom do different work from ours. The comparison shows what users are used to. It is not a target we can match.
- **The speed gains are untested.** Every estimate of faster answers in this plan is a rough estimate from published guidance. None was measured on our system.
- **Quality is the limit.** This chatbot answers questions about financial data. A faster but less accurate answer is a worse result. Every change in Stages 2 and 3 must pass the evaluation suite.

---

## Recommended timeline

| Week | Action | Outcome |
|---|---|---|
| 1 | Deploy step timing. Check the model name. Check the cache. | We know the model and the cache status. Timing data starts to arrive. |
| 2 | Collect data. Agree the progress-message design with the front-end team. | At least 200 requests with step times. |
| 3 to 4 | Build progress messages. Test thinking level and answer length. | Progress messages ready for review. First quality results. |
| 5 to 6 | Release progress messages. Choose the settings that keep quality. | Users see progress. We know the speed gain of the new settings. |
| 7 to 10 | Test a faster model and conditional search. | Decision on each, with quality results. |
| Later | Plan word-by-word answers. | Scope and effort estimate. |

---

## What success looks like

The values in the last two columns are targets that we propose. They are not forecasts. We will confirm them after the Stage 1 data.

| Measurement | Today | After 4 weeks | After 3 months |
|---|---|---|---|
| Time to first sign of progress | No sign until the full answer is ready (10 to 58 seconds) | About 1 second | About 1 second |
| Typical answer (median) | 10.4 seconds | No worse than today | 10 seconds or less |
| Slowest 5% of answers (95th percentile) | 32.3 seconds | Under 30 seconds, and no request over 60 seconds | Under 20 seconds |
| Requests ended by the load balancer (60-second limit) | Not tracked in this plan; the slowest October answers were close to the limit | Zero | Zero |
| Answer quality (evaluation suite) | Current score | No drop | No drop |

---

## What we need from stakeholders

- **Engineering time:** 1 backend engineer, part-time, for about 6 to 10 weeks (rough estimate).
- **Front-end team time:** About 1 to 2 weeks to show progress messages in the apps that call the chatbot (rough estimate). We need to know which team owns this.
- **Product input:** A decision on how much answer quality we may trade for speed (see question 2).
- **Leadership cadence:** A short review after Stage 1 and after Stage 2, 30 minutes each.

---

## Questions for decision-makers

1. Is the wait that users feel (first sign of progress) the main goal, or is total answer time the main goal? Progress messages help the first one most.
2. If a faster setting gives slightly shorter answers, but the evaluation score stays the same, do we accept it?
3. Who owns the front end that shows the chatbot, and can that team show progress messages in the next 4 weeks?
4. Do we allow tests of AI models from other providers, or do we stay with Gemini?
5. Is the answer cache meant to be on in production? If not, why not?

---

## Related documents

- `docs/chatbot-latency-industry-benchmarks.md` (full benchmark research and sources)
- `docs/adr/0001-fix-event-loop-blocking-llm-calls.md` and `docs/adr/0003-native-async-gemini-client.md` (earlier speed work)
- `loadtest/README.md` (load test method and the 60-second limit)
