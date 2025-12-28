# State-Managed Insurance System (Core 1–Core 9)

## 1. 프로젝트 개요

본 프로젝트는 개인 단위 헬스케어 시계열 데이터를 기반으로, 보험 의사결정 시스템에서 안정성이 예측 정확도(Accuracy, AUC, F1 등)가 아니라 **의사결정 구조(Decision Layer)**에서 발생한다는 가설을 단계적으로 검증하는 것을 목표로 합니다.

일반적인 헬스케어/보험 분석은 모델 성능 지표를 개선하는 방향으로 수렴하는 경우가 많습니다. 그러나 운영 관점에서는 예측이 존재하더라도 개입 판단이 시간축에서 빈번히 바뀌면(toggling) 시스템 행위 자체가 불안정해집니다. 본 프로젝트는 **동일한 예측(동일 입력/동일 로그)을 고정한 상태에서 decision layer만 바꾸었을 때, 개입의 안정성 지표가 구조적으로 달라지는지**를 검증합니다.

이를 위해 Core 1부터 Core 9까지 점진적으로 문제를 확장하였으며, 예측 가능성 확보 → 규칙 기반 결정의 불안정 노출 → 상태 계측(μHSM) 도입 → 상태 기반 재판단(Core 9) → 안정성 지표 비교(Core 7)로 구성합니다. 또한 제출용 저장소에서는 notebooks 중심 설명이 아니라 **scripts 기반 파이프라인 + Docker(MLflow) + CI(GitHub Actions) + Cloud(AWS EC2 선택 적용)**로 “구조 실행 가능”을 증명하도록 설계합니다.

본 프로젝트는 다음 사항을 명확히 지양합니다.

- 임상적 효과 또는 의료 성능에 대한 주장  
- 보험 상품 성과(손해율 개선 등)에 대한 주장  
- 특정 ML 모델 성능 우열 비교(정확도 경쟁)

대신, **의사결정 구조의 일반성**, **재실행 가능한 파이프라인**, **안정성 지표 기반 비교**, **시스템 설계 관점의 확장성**에 초점을 둡니다.

---

## 2. Core 1–Core 9 전체 구성 요약

| Core | 주제 | 핵심 질문 |
|---|---|---|
| Core 1 | Prediction Feasibility | 개인 상태 변화는 예측 가능한 대상인가? |
| Core 2 | Rule-based Insurance Decision | 예측을 규칙에 직접 연결하면 의사결정이 안정되는가? |
| Core 3 | Information Density / Uncertainty | 예측값만으로 판단할 때 왜 결정이 흔들리는가? |
| Core 4 | Log & DB Systemization | 행동이 상태를 안정화했는지를 검증 가능한 구조로 만들 수 있는가? |
| Core 5 | Prediction-based Decision Log | 예측 기반 decision layer를 고정 산출물로 만들 수 있는가? |
| Core 6 | μHSM Construction | 점수(point)가 아니라 상태(state)를 계측할 수 있는가? |
| Core 7 | Decision Stability Test | decision layer 변경만으로 안정성이 개선되는가? |
| Core 8 | Structural Interpretation (md) | 왜 이런 결과가 나올 수밖에 없는가를 구조적으로 설명할 수 있는가? |
| Core 9 | State-based Re-decision | μHSM 기반 재판단을 파이프라인으로 고정할 수 있는가? |

각 Core는 이전 단계의 결과를 전제로 하여 확장되며, 단일 Core만으로 독립적인 결론을 주장하지 않습니다.

---

## 3. 프로젝트 디렉터리 구조

본 저장소의 주요 파일 구조는 다음과 같습니다.

| 경로 | 설명 |
|---|---|
| `notebooks/` | Core 단위 노트북(해석 및 설계 확인용) |
| `scripts/` | 제출용 재실행 파이프라인(검증의 중심) |
| `scripts/run_core5.py` | Core5 decision log 생성(포맷/검증/저장/중복 방지) |
| `scripts/run_core9.py` | μHSM merge + Core9 재판단 log 생성(중복 방지 포함) |
| `scripts/run_compare.py` | Core5 vs Core9 안정성 비교(summary 생성) |
| `scripts/run.py` | Core5 → Core9 → Compare 순차 실행 엔트리 |
| `data/derived/` | 분석 및 산출물 CSV 디렉터리(입출력 기준 고정) |
| `docker/` | Dockerfile / docker-compose(MLflow) |
| `.github/workflows/` | GitHub Actions CI 파이프라인 검증 |
| `core8_structural_interpretation.md` | 구조 해석 문서 |

각 노트북은 **시뮬레이션 재실행이 아닌 결과 해석 및 구조 설명**을 목적으로 작성되며, 제출용 재현은 `scripts/`에서 고정합니다.

---

## 4. Core 1 — Prediction Feasibility

Core 1에서는 이후 단계에서 문제가 발생했을 때 원인이 “예측 실패”인지 “의사결정 구조 실패”인지 분리하기 위한 사전 검증을 수행합니다. 따라서 Core 1의 목표는 최고 성능 모델을 만드는 것이 아니라, **개인 상태 변화가 최소 수준에서라도 예측 가능한 구조를 가진다는 전제**를 확보하는 데 있습니다.

본 단계에서는 다음 관점을 고정합니다.

- 상태(state)를 단일 라벨이 아니라 시간축에서 연속적으로 변하는 값으로 다룹니다.  
- 예측 대상은 정답 분류가 아니라 **단기 상태 변화(Δstate)** 또는 단기 상태 흐름으로 정의합니다.  
- 예측 가능성이 확인되면, 이후 단계에서 흔들림이 관찰되더라도 이를 “모델이 못 맞췄다”로 축소할 수 없도록 합니다.

Core 1의 결론은 “예측이 된다/안 된다”의 단순 결론이 아니라, **이후의 핵심 문제가 decision layer로 이동하도록 설계를 정당화하는 전제**로 기능합니다.

---

## 5. Core 2 — Rule-based Insurance Decision

Core 2에서는 보험 시스템에서 흔히 사용되는 전형적인 구조를 구현합니다.  
- 예측 결과를 임계값(threshold)과 비교합니다.  
- 임계값을 넘으면 개입(intervention)합니다.  
- 임계값을 넘지 않으면 개입하지 않습니다.  

이 구조는 단순하지만, 운영 관점에서 다음 문제가 발생할 수 있습니다.  
- 동일한 asset에서 개입/미개입이 시간축으로 반복 전환되는 **토글(toggling)**이 발생합니다.  
- 예측이 경계값 근처에서 흔들리면 개입 여부가 스위치처럼 빈번히 바뀝니다.  
- 결과적으로 “예측이 존재한다”는 사실만으로 의사결정이 안정화된다고 보기 어렵습니다.  

Core 2의 핵심 의미는 다음과 같습니다.  
- 불안정의 원인을 모델 성능으로만 설명하기 어려우며, 결정 구조 자체가 흔들림을 만들어낼 수 있음을 드러냅니다.  
- 같은 예측을 넣더라도 decision layer 설계에 따라 시스템 행위(개입 패턴)가 달라질 수 있다는 전제가 형성됩니다.  

---

## 6. Core 3 — Information Density / Uncertainty

Core 3에서는 Core 2에서 관찰된 토글과 오개입 가능성을 “성능 부족”이 아니라 **의사결정 입력의 정보 밀도 부족**으로 해석합니다. 즉, 점수 하나로 개입을 스위치처럼 결정할 때 발생하는 문제를 구조적으로 설명하고, 이후 설계(μHSM)의 필요성을 고정합니다.

본 단계의 핵심은 다음을 명확히 하는 것입니다.

- 동일한 예측값이라도 “그 예측을 얼마나 믿어야 하는지” 정보가 없으면 결정이 폭주할 수 있습니다.  
- 예측의 작은 진동이 곧바로 개입/미개입 전환으로 연결되면 토글이 구조적으로 유발됩니다.  
- 따라서 decision layer 입력은 단일 point가 아니라 **불확실성/관측 가능성/국면 정보**를 포함하는 형태로 확장되어야 합니다.

Core 3는 “불확실성을 잘 추정했다”가 아니라, **왜 상태 기반 입력이 필요해지는지**를 Core 2의 실패와 연결해 설명하는 단계입니다.

---

## 7. Core 4 — Log & DB Systemization

Core 4는 모델 성능을 올리는 단계가 아니라, 이후 모든 검증을 가능하게 만드는 **로그 기반 시스템 구조화 단계**입니다. Core 5–9에서 반복적으로 확인해야 하는 질문은 “개입이 안정화로 이어졌는가”이며, 이는 단발성 출력이 아니라 **시간축 로그**로만 검증 가능합니다.

Core 4에서는 다음을 고정합니다.

- 상태(state), 예측(prediction), 행동(action)을 분리된 로그로 관리합니다.  
- key를 `asset_id`, `date`, `t_index`로 통일하여 join이 붕괴되지 않도록 설계합니다.  
- 결합 안정성(중복 폭발/시간축 불일치)을 방지하여, 이후 Core 7 비교가 “같은 입력” 위에서 성립하도록 합니다.

Core 4의 산출은 결론이 아니라 **검증의 기반**이며, 이 단계가 없으면 Core 5의 decision log 자체가 제출 가능한 형태로 고정되지 않습니다.

---

## 8. Core 5 — Prediction-based Decision Log

Core 5에서는 예측 기반 decision layer를 “제출용 기준선(baseline)”으로 고정합니다. 이 단계의 핵심 산출물은 모델 결과가 아니라 **시간축 의사결정 로그(CSV/DB)**이며, 이후 Core 9와 안정성 비교가 가능하도록 포맷을 고정합니다.

Core 5에서는 다음을 수행합니다.

- 예측 기반 규칙으로 개입 여부를 결정하고, 그 결과를 `data/derived/core5_decision_log.csv`로 저장합니다.  
- 동일 입력이면 동일 로그가 생성되도록 실행 경로, 컬럼, key 구조를 고정합니다.  
- 개입 이후 상태 변화(안정화 여부)를 동일 기준으로 라벨링하여, “개입이 의미 있었는지”를 비교 가능하게 만듭니다.

Core 5의 핵심 관찰 대상은 다음과 같습니다.

- 토글이 얼마나 발생하는지(개입 플래그의 빈번한 전환)  
- 오개입이 얼마나 존재하는지(개입했지만 안정화로 이어지지 않음)  
- 예측이 존재하더라도 예측 기반 규칙이 안정성을 보장하지 않는다는 사실을 baseline으로 고정합니다.

---

## 9. Core 6 — μHSM Construction

Core 6에서는 “더 좋은 예측 모델”을 만드는 대신, decision layer 입력을 **점(point)에서 상태(state)로 확장**하기 위한 μHSM(마이크로 Health State Monitor) 구조를 구성합니다. Core 3에서 고정한 “정보 밀도 부족” 문제를 해소하기 위한 설계 단계입니다.

μHSM은 다음 요소를 상태 벡터로 포함합니다.

- `state_value`(현재 상태), `degradation_rate`(악화/회복 변화율)  
- `HSI`, `HDR`(상태 지표 성분)  
- `recovery_margin`(회복 여력)  
- `observability_score`(관측 가능성/신뢰도)  

Core 6의 핵심은 다음과 같습니다.

- 단일 점수 기반 규칙은 경계값 부근에서 토글이 필연적으로 발생할 수 있습니다.  
- 상태 기반 입력은 “현재 국면이 무엇인지”를 decision layer에 제공하여, 불확실한 구간에서 섣부른 전환을 줄이는 방향으로 작동합니다.  
- 따라서 μHSM은 모델 교체가 아니라 **판단 입력 구조의 확장**이며, Core 9 재판단의 기반이 됩니다.

---

## 10. Core 7 — Decision Stability Test

Core 7에서는 모델 성능 지표(Accuracy/AUC)를 사용하지 않고, 오직 **의사결정 안정성 지표**로 Core 5와 Core 9를 비교합니다. 이 단계는 “좋아 보인다”가 아니라, 제출용 레포에서 **정량 비교의 기준을 고정**하는 역할을 합니다.

Core 7은 다음 절차로 구성합니다.

- Core 5 decision log를 로드합니다.  
- μHSM 상태 모니터(derived)를 병합하여, 동일 key 기준의 비교 기반을 만듭니다.  
- Case A(Core 5)와 Case B(Core 9)를 동일 지표로 평가합니다.

평가 지표는 아래 3개로 제한합니다.

- `toggle_rate`: 시간축에서 개입 플래그가 얼마나 자주 바뀌는지  
- `false_intervention`: 개입했지만 안정화로 이어지지 않은 비율/건수  
- `stabilization_rate`: 개입 후 안정화로 이어진 비율  

Core 7의 핵심은 “예측을 고정해도 decision layer가 달라지면 시스템 행위가 달라진다”를 안정성 지표로 증명하는 데 있습니다.

---

## 11. Core 8 — Structural Interpretation (core8_structural_interpretation.md)

Core 8은 코드나 수치가 아니라, 결과가 그렇게 나올 수밖에 없는 이유를 **구조적으로 해석해 문서로 고정**하는 단계입니다. 제출용 레포에서 Core 8은 “실험 결과 요약”이 아니라, 설계 의도가 일관되게 이어졌음을 설명하는 근거 문서입니다.

Core 8에서는 다음을 고정합니다.

- 규칙 기반(point 기반) 의사결정이 시간축에서 흔들릴 수밖에 없는 구조적 이유를 설명합니다.  
- μHSM이 “정확도를 올린다”가 아니라 “판단 입력을 상태로 바꾼다”는 구조적 차이를 명확히 합니다.  
- 상태 기반 재판단이 토글/오개입을 줄이는 방향으로 작동하는 이유를 국면/관측 가능성 관점에서 정리합니다.

Core 8은 결과를 과장하지 않으며, **왜 decision layer 비교가 의미 있는지**를 제출 문서로 고정합니다.

---

## 12. Core 9 — State-based Re-decision (μHSM 기반 재판단)

Core 9는 “좋은 모델을 새로 만든다”가 아니라, Core 5에서 이미 생성된 예측/결정 로그를 유지한 채 **decision layer만 상태 기반으로 재구성**하는 단계입니다. 즉, 예측 계층은 바꾸지 않고 판단 계층만 교체하여, 개입의 정답이 아니라 **개입의 안정성(토글/오개입/안정화)**이 어떻게 달라지는지 검증합니다.

Core 9의 구성은 다음과 같습니다.

- 입력은 `data/derived/core5_decision_log.csv`와 μHSM derived(`data/derived/muHSM_state_monitor.csv` 등)로 고정합니다.  
- key는 `asset_id`, `date`, `t_index`로 통일하여 병합이 가능한 구조를 유지합니다.  
- 병합 과정에서는 key 기준 dedup과 결합 후 row 수 검증을 수행하여, 중복 폭발로 비교가 깨지는 상황을 방지합니다.

Core 9의 산출물은 다음 포맷으로 고정합니다.

- `data/derived/core9_state_based_decision_log.csv`  
- `asset_id,date,t_index,state_value,degradation_rate,HSI,HDR,recovery_margin,observability_score,intervention_flag_core9,stabilized`

Core 9의 핵심은 다음과 같습니다.

- Core 5와 동일한 예측/로그 기반에서 decision layer만 바꾸어 비교 가능성을 유지합니다.  
- 상태 벡터(국면/변화율/관측 가능성)를 이용해, 경계값 진동에 의한 스위치형 결정을 줄이는 방향으로 재판단합니다.  
- scripts 재실행 시 동일 입력이면 동일 출력이 생성되도록 파이프라인을 고정하여, 제출용 레포에서 “구조 실행 가능”을 증명합니다.

---

## 13. 실행 환경

| 항목 | 환경 |
|---|---|
| 실행 환경 | 로컬 macOS (Python 3.10) + EC2(Ubuntu) 선택 적용 |
| 데이터베이스 | MySQL (중복 적재 방지 설계 포함) |
| 주요 라이브러리 | pandas, numpy, sqlalchemy |
| 실험 기록 | MLflow (Docker compose 기반) |
| 컨테이너 | Docker (MLflow base image 기반) |
| CI | GitHub Actions (push 트리거 자동 검증) |

---

## 14. 프로젝트의 역할 정의

본 프로젝트는 특정 모델의 정확도를 올리는 경쟁을 목표로 하지 않습니다. 대신, 동일한 예측이 존재하는 상황에서 **의사결정 구조가 시스템 안정성을 어떻게 바꾸는지**를 구조적으로 설명하고, 재현 가능한 파이프라인으로 고정하는 것을 목표로 합니다.

이를 위해 본 프로젝트는 다음을 수행합니다.

- Core 5에서 예측 기반 규칙의 decision log를 baseline으로 고정합니다.  
- Core 6에서 μHSM을 통해 판단 입력을 점수에서 상태로 확장합니다.  
- Core 7에서 안정성 지표(`toggle_rate`, `false_intervention`, `stabilization_rate`)로 구조 비교를 수행합니다.  
- Core 8에서 결과가 필연적으로 형성되는 이유를 해석 문서로 고정합니다.  
- Core 9에서 동일 예측 위에서 decision layer만 교체한 재판단 파이프라인을 재실행 가능하게 고정합니다.

결론적으로 본 프로젝트는 “예측 결과”가 아니라 “의사결정 구조”가 시스템 행위를 결정한다는 점을 Core 1–Core 9의 단계적 설계와 제출용 파이프라인(scripts/Docker/CI/Cloud 선택 적용)으로 고정한 사례입니다.
