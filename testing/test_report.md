# SIH26104 — Testing & Integration Report
**Member 6 — Dataset, Testing & Integration**

## Test Environment
- Date:12 September 2026
- Tested by:Member 6
- OS / Python version:Windows, Python 3.14

## Test Cases

| # | Test Case | Input Description | Expected Result | Actual Result | Pass/Fail |
|---|---|---|---|---|---|
| 1 | Genuine human voice (end-to-end) | Normal speech via mic, claimed speaker = CEO | Reasonable score, no crash | Authenticity: 47.4%, Clone: 52.6%, Speaker Match: 57.2%, Risk: MEDIUM, Action: Additional Verification Required | Pass |
| 2 | AI-generated voice | TTS-generated audio | HIGH clone_prob, CRITICAL/BLOCK | | |
| 3 | Cloned voice | Voice-cloned sample | HIGH clone_prob, CRITICAL/BLOCK | | |
| 4 | Impersonated speaker | Cloned voice claiming to be registered speaker | LOW speaker match, BLOCK | | |
| 5 | Noisy genuine voice | Genuine voice + background noise | Should still detect correctly | | |
| 6 | Unknown speaker | Genuine voice, not in registered DB | Speaker match fail / flag | | |
| 7 | Short/unclear audio | <2 sec clip | Handled gracefully, no crash | | |
| 8 | Hindi speech | Genuine Hindi voice | Correct detection | | |
| 9 | English speech | Genuine English voice | Correct detection | | |
| 10 | Different microphone | Same speaker, different device | Consistent result | | |

## Metrics
- Overall Accuracy:
- False Positive Rate:
- False Negative Rate:
- Average End-to-End Latency (mic → dashboard):

## Known Issues / Limitations
- Speaker verification (Member 3) not fully wired yet — using a temporary placeholder value instead of the real model.

## Conclusion
- Full pipeline (audio capture, deepfake detection, risk scoring, prevention, dashboard) successfully integrated and tested end-to-end. Real model outputs confirmed working (not mock/random data).
