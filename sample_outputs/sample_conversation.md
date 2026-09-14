# Sample Conversation — Enterprise Knowledge Assistant

Configuration used for this sample run: **LLM Provider = OpenAI (gpt-4o-mini)**,
**Vector Store = FAISS**.

---

### Turn 1 — Basic retrieval + source citation

**User:** What is the leave policy?

**Assistant:**
> NimbusTech offers several types of leave:
> - **Earned Leave (EL):** 1.5 days/month (18 days/year), applied for at least 3 working days in advance.
> - **Casual Leave (CL):** 12 days/year, max 3 consecutive days at a time.
> - **Sick Leave (SL):** 10 days/year; more than 2 consecutive days requires a medical certificate.
> - **Maternity Leave:** 26 weeks paid (first two children), 12 weeks from the third child onward.
> - **Paternity Leave:** 10 working days, to be availed within 3 months of the child's birth.
> - **Bereavement Leave:** up to 5 days paid.

**Sources:** `Leave_Policy.pdf`

---

### Turn 2 — Conversational memory (follow-up question)

**User:** What about carry-forward?

*(Internally, the system condenses this into a standalone question:
"What is the carry-forward rule for earned leave at NimbusTech?", correctly
resolving "carry-forward" back to the leave policy discussed in Turn 1.)*

**Assistant:**
> Up to 10 days of unused Earned Leave can be carried forward to the next
> calendar year. Any Earned Leave beyond 10 days that remains unused by
> December 31 will lapse, unless you've opted for encashment. Note that
> Casual Leave and Sick Leave cannot be carried forward — they lapse at the
> end of the calendar year.

**Sources:** `Leave_Policy.pdf`

---

### Turn 3 — Cross-document retrieval

**User:** How much daily allowance do I get on international travel, and what should I do if I lose my laptop while traveling?

**Assistant:**
> For international business travel, you're entitled to a per-diem of
> **USD 60 per day** for meals and incidentals (this doesn't apply when
> meals are provided by the client or hotel).
>
> If you lose your laptop while traveling, report it to IT Security within
> **1 hour** of discovery by emailing security@nimbustech.example.com or
> calling the IT Helpdesk, so the device can be secured/remotely wiped.

**Sources:** `Travel_Policy.docx`, `IT_Policy.pdf`

---

### Turn 4 — Hallucination handling (out-of-scope question)

**User:** What is the company's stock price target for next quarter?

**Assistant:**
> I could not find this information in the available company documents.

**Sources:** *(none shown — answer was not grounded in retrieved context)*

---

### Turn 5 — FAQ + benefits retrieval

**User:** What health insurance coverage do I get, and can I add my parents?

**Assistant:**
> You're covered under NimbusTech's Group Mediclaim Insurance, which
> includes the employee, spouse, and up to 2 dependent children, with a
> base sum insured of **INR 5,00,000** per family per year. A top-up plan
> (additional INR 5,00,000) is available at a subsidized premium.
>
> Yes — parents/parents-in-law can be added under a **voluntary parental
> cover** at your own cost, with a sum insured of **INR 3,00,000**.

**Sources:** `Benefits_Documentation.md`

---

## Notes on this transcript

- Turns 1–2 demonstrate **conversational memory**: the pronoun-like
  follow-up "What about carry-forward?" is correctly resolved without the
  user having to repeat "leave policy".
- Turn 3 demonstrates **hybrid search + reranking** surfacing the correct
  chunks from two different source documents for a single compound question.
- Turn 4 demonstrates **hallucination handling**: the assistant explicitly
  states the information isn't available rather than fabricating an answer,
  and no (potentially misleading) sources are displayed.
- Every grounded answer lists its **source document(s)** exactly as they
  appear in `data/raw/`, matching the functional requirement's example format.
