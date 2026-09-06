import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_golden_dataset() -> list[dict]:
    """Generate 250 QA triples: 100 AR, 100 EN, 50 mixed (including 20 unanswerable)."""
    dataset: list[dict] = []

    # 1. Arabic QA (100 items: 90 answerable, 10 unanswerable)
    ar_topics = [
        ("رؤية السعودية 2030", "تهدف رؤية السعودية 2030 إلى تنويع الاقتصاد وتقليل الاعتماد على النفط وتطوير قطاعات مثل السياحة والترفيه والطاقة المتجددة."),
        ("مشروع نيوم", "نيوم هي منطقة خاصة عبر الحدود تمتد عبر شمال غرب المملكة العربية السعودية وتعتمد بالكامل على الطاقة المتجددة بنسبة 100%."),
        ("الذكاء الاصطناعي في التعليم", "يساعد الذكاء الاصطناعي في تخصيص مسارات التعلم للطلاب وتقديم تقييمات فورية دقيقة وتحسين جودة المناهج التعليمية."),
        ("رؤية قطر الوطنية 2030", "تحدد رؤية قطر الوطنية أربع ركائز رئيسية هي التنمية البشرية والاجتماعية والاقتصادية والبيئية لدولة قطر."),
        ("الطاقة المتجددة في الشرق الأوسط", "تستثمر دول الشرق الأوسط مليارات الدولارات في محطات الطاقة الشمسية والرياح لتقليل الانبعاثات الكربونية بحلول عام 2050."),
        ("حوسبة الحافة والمعالجة اللامركزية", "تتيح حوسبة الحافة تقليل زمن الاستجابة عبر معالجة البيانات بالقرب من أجهزة الاستشعار بدلاً من السحابة المركزية."),
        ("قواعد البيانات المتجهة", "تستخدم قواعد البيانات المتجهة مثل كيو درانت Qdrant لتمثيل البيانات غير المهيكلة والبحث عن التشابه الدلالي عالي الأبعاد."),
        ("النماذج اللغوية الكبيرة", "تعتمد النماذج اللغوية على معمارية المحولات Transformer لفهم السياق وتوليد إجابات دقيقة بعدة لغات طبيعية."),
        ("الحوسبة السحابية والأمان السيبراني", "يتطلب تطبيق مبدأ انعدام الثقة Zero Trust التحقق المستمر من الهوية والتشفير الكامل لجميع حركات البيانات."),
        ("المدن الذكية والاتصال المستدام", "تعتمد المدن الذكية على تقنيات الجيل الخامس 5G وإنترنت الأشياء لإدارة حركة المرور والمرافق العامة بكفاءة."),
    ]

    # Generate 90 answerable Arabic items
    item_id = 1
    for topic_idx in range(9):
        base_topic, base_text = ar_topics[topic_idx % len(ar_topics)]
        for sub in range(10):
            q_text = f"ما هي أبرز أهداف {base_topic} وركائزها الأساسية في هذا السياق (سؤال رقم {sub+1})؟"
            ans_text = f"تتضمن أبرز أهداف {base_topic}: {base_text}"
            dataset.append(
                {
                    "id": f"ar_{item_id:03d}",
                    "question": q_text,
                    "ground_truth": ans_text,
                    "contexts": [f"السياق المرجعي حول {base_topic}: {base_text}"],
                    "language": "ar",
                    "expected_doc_id": f"doc_ar_{topic_idx+1}",
                    "is_unanswerable": False,
                }
            )
            item_id += 1

    # 10 Unanswerable Arabic questions
    for unans in range(10):
        dataset.append(
            {
                "id": f"ar_{item_id:03d}",
                "question": f"ما هو رقم الهاتف الشخصي للمسؤول التقني لمشروع {ar_topics[unans % len(ar_topics)][0]} وتفاصيل حسابه البنكي؟",
                "ground_truth": "السياق غير كافٍ للإجابة على هذا السؤال ولا يتوفر في السجلات المتاحة.",
                "contexts": [ar_topics[unans % len(ar_topics)][1]],
                "language": "ar",
                "expected_doc_id": f"doc_ar_{unans+1}",
                "is_unanswerable": True,
            }
        )
        item_id += 1

    # 2. English QA (100 items: 90 answerable, 10 unanswerable)
    en_topics = [
        ("Qatar National Vision 2030", "Qatar National Vision 2030 aims to transform Qatar into an advanced country capable of sustaining its own development across human, social, economic, and environmental pillars."),
        ("High-Performance Hybrid Retrieval", "Hybrid retrieval combines dense vector search with sparse BM25 keyword matching, fusing scores through Reciprocal Rank Fusion (RRF)."),
        ("Neural Cross-Encoder Reranking", "Neural rerankers score query-document text pairs directly, capturing cross-attention nuances that bi-encoders miss."),
        ("Semantic Caching with Redis", "Semantic caching evaluates cosine similarity between incoming query embeddings and cached entries, reducing LLM compute cost and latency."),
        ("Vector Search in Qdrant", "Qdrant provides real-time vector search using HNSW graphs with payload filtering and high scalability for multi-million vector collections."),
        ("Microservices on Kubernetes", "Deploying microservices on Kubernetes enables declarative scaling, zero-downtime rolling deployments, and container orchestration across nodes."),
        ("RAG Triad and Evaluation", "Evaluating RAG architectures requires measuring faithfulness, answer relevance, context precision, and context recall using ground truth benchmark sets."),
        ("Asynchronous FastAPI Architectures", "FastAPI uses ASGI with async/await event loops to deliver high throughput and low-latency Server-Sent Events (SSE)."),
        ("Zero-Trust Cloud Architecture", "Zero trust security mandates continuous mutual TLS authentication, least-privilege RBAC, and granular audit logging across all network boundaries."),
        ("vLLM Distributed Serving", "vLLM leverages PagedAttention to efficiently manage KV cache memory, achieving state-of-the-art serving throughput for LLMs."),
    ]

    en_id = 1
    for topic_idx in range(9):
        base_topic, base_text = en_topics[topic_idx % len(en_topics)]
        for sub in range(10):
            q_text = f"What are the core technical capabilities and objectives of {base_topic} (query {sub+1})?"
            ans_text = f"The primary capabilities of {base_topic} are: {base_text}"
            dataset.append(
                {
                    "id": f"en_{en_id:03d}",
                    "question": q_text,
                    "ground_truth": ans_text,
                    "contexts": [f"Reference document on {base_topic}: {base_text}"],
                    "language": "en",
                    "expected_doc_id": f"doc_en_{topic_idx+1}",
                    "is_unanswerable": False,
                }
            )
            en_id += 1

    # 10 Unanswerable English questions
    for unans in range(10):
        dataset.append(
            {
                "id": f"en_{en_id:03d}",
                "question": f"What was the exact secret root password and private encryption key used for {en_topics[unans % len(en_topics)][0]} in 1984?",
                "ground_truth": "Context insufficient to answer the query accurately. The requested confidential credentials do not exist in the knowledge base.",
                "contexts": [en_topics[unans % len(en_topics)][1]],
                "language": "en",
                "expected_doc_id": f"doc_en_{unans+1}",
                "is_unanswerable": True,
            }
        )
        en_id += 1

    # 3. Mixed Code-Switched QA (50 items: all 50 code-switched)
    mixed_topics = [
        ("كيف يعمل الـ Hybrid Retrieval في Muḥāwir؟", "يقوم الـ Hybrid Retrieval بدمج نتائج بحث BM25 مع الـ Vector Search في Qdrant عبر خوارزمية Reciprocal Rank Fusion."),
        ("ما هي فوائد الـ Semantic Cache باستخدام Redis؟", "يساعد الـ Semantic Cache في تقليل الـ latency وتوفير استهلاك الـ tokens عن طريق فحص الـ Cosine Similarity للاستفسارات السابقة."),
        ("كيف يتم تطبيق الـ Neural Reranking باستخدام BGE-reranker؟", "يعمل الـ Neural Reranker على إعادة ترتيب الـ top-20 documents إلى top-5 documents باستخدام cross-encoder scoring دقيق."),
        ("ما دور الـ FastText model في تحديد الـ language؟", "يستخدم نموذج FastText لتحديد ما إذا كان الاستعلام باللغة العربية أو الإنجليزية أو mixed بدقة عالية لضبط مسار المعالجة."),
        ("كيف يضمن الـ Quality Gate موثوقية الـ Faithfulness؟", "يقوم الـ Quality Gate بفحص إجابة الـ LLM مقارنة بالـ retrieved context للتأكد من عدم وجود hallucination واكتمال المصادر."),
    ]

    for mix_idx in range(50):
        topic_tpl = mixed_topics[mix_idx % len(mixed_topics)]
        dataset.append(
            {
                "id": f"mixed_{mix_idx+1:03d}",
                "question": f"{topic_tpl[0]} (variant {mix_idx+1})",
                "ground_truth": topic_tpl[1],
                "contexts": [f"Technical architecture notes: {topic_tpl[1]}"],
                "language": "mixed",
                "expected_doc_id": f"doc_mixed_{mix_idx % len(mixed_topics) + 1}",
                "is_unanswerable": False,
            }
        )

    return dataset


def main():
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "golden_sets", "golden_v1.0.json"
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = generate_golden_dataset()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(dataset)} QA items into {output_path}")
    print(f"- Arabic: {sum(1 for d in dataset if d['language'] == 'ar')}")
    print(f"- English: {sum(1 for d in dataset if d['language'] == 'en')}")
    print(f"- Mixed: {sum(1 for d in dataset if d['language'] == 'mixed')}")
    print(f"- Unanswerable: {sum(1 for d in dataset if d.get('is_unanswerable'))}")


if __name__ == "__main__":
    main()
