# Contributing to Nusantara Fire Sentinel — ML Worker

Thank you for your interest in the **Nusantara Fire Sentinel** Heavy-Compute ML Worker project!

[🇮🇩 Baca Panduan Kontribusi dalam Bahasa Indonesia](CONTRIBUTING-IN.md)

---

## 📜 Open Source & Forking Rights (MIT License)

This project is licensed under the permissive **[MIT License](LICENSE)**.

* **Feel free to Fork & Build**: Any researcher, machine learning engineer, developer, or organization is completely free to fork this repository, adapt model weights, customize feature engineering pipelines, and deploy private or public inference instances under the terms of the MIT License.
* **Commercial and Non-Commercial Use**: You can use, modify, experiment with, and distribute this codebase on your own forks without paying royalties or requiring formal approval, provided the original copyright notice and license are preserved.

---

## 🛡️ Upstream Contribution Policy: Curated / Invitation-Only Model

While the software is fully open source to fork and modify, the **official upstream repository** (`ayusuf29/karthutla-ml-worker`) operates under a **Curated and Selective Contribution Model**:

### Why is direct contribution restricted on the upstream repo?
1. **Critical ML Inference Reliability**: The ML Worker powers real-time wildfire ignition risk scoring and SAR Sentinel-1 radar anomaly classification for national peatland protection. Numerical stability, feature alignment (PP 57/2016 peat regulations), and strict gRPC protobuf contract compatibility are critical.
2. **Hardware Acceleration Constraints**: The production pipeline is tailored for AMD Cloud ROCm accelerators (AMD Instinct MI300X/MI250/MI210 and Radeon Pro GPUs) with fallback to CPU execution. Unvetted dependencies or memory management changes can cause hardware GPU kernel regressions.
3. **Selective PR & Issue Processing**: To maintain deterministic model behavior and protect maintainer bandwidth, **direct Pull Requests, Issue creation, and Merge permissions are restricted to the core maintainer and approved collaborators**.

---

## 🤝 How to Propose Changes or Collaborate

If you have developed model improvements, faster gRPC serialization, or new remote sensing feature extractors on your fork:

1. **Fork the Repository**: Clone and build your improvements on your own GitHub fork.
2. **Validate Thoroughly Locally**: Ensure the service starts, `compile_proto.py` succeeds, and `test_client.py` reports clean evaluation metrics against the gRPC service.
3. **Contact the Maintainer**: Reach out directly to the core maintainer before opening a Pull Request:
   * **Maintainer**: Achmad Yusuf Ramadhan
   * **LinkedIn**: [https://linkedin.com/in/cupu2903](https://linkedin.com/in/cupu2903)
   * **GitHub**: [@ayusuf29](https://github.com/ayusuf29)
4. **Invitation / Review Process**: After technical alignment, the maintainer will either grant collaborator status or invite you to submit a designated PR for code review and merge.

> **Note**: Unsolicited PRs or automated bot PRs opened on the upstream repository without prior coordination may be reviewed selectively or closed without review.

---

## 💻 Technical Standards for Collaborators

For approved collaborators working on upstream branches or PRs:

* **Python & Dependencies**: Python 3.10+, strict adherence to `requirements.txt` versions, avoiding heavy unpinned dependencies.
* **Protobuf & gRPC**: Never introduce breaking schema modifications to `proto/fire_risk_service.proto` without updating both client and server stubs simultaneously.
* **Model Serialization**: Keep model artifacts cleanly versioned in `models/` with documented evaluation benchmarks.
* **Security & Data**: Never commit proprietary satellite raw dumps, confidential geospatial credentials, or unauthorized access tokens into git.

Thank you for supporting scientific open-source technology for forest and peatland preservation!
