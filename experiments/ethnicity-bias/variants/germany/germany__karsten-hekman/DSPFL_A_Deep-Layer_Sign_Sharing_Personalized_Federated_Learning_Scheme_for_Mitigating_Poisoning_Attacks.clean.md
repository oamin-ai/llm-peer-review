# DSPFL A Deep-Layer Sign Sharing Personalized Federated Learning Scheme for Mitigating Poisoning Attacks

Author: Karsten Hekman
University: Pennsylvania State University


## Abstract

With the rise of the smart industry, machine learning (ML) has become a popular method to improve the security of the Industrial Internet of Things (IIoT) by training anomaly detection models. Federated learning (FL) is a distributed ML scheme that facilitates anomaly detection on IIoT by preserving data privacy and breaking data silos. However, poisoning attacks pose significant threats to FL, where adversaries upload poisoned local models to the aggregation server, thereby degrading model accuracy. The prevalence of non-independent and identically distributed (non-IID) data across IIoT devices further exacerbates this threat, as it naturally leads to diverse local models, making malicious ones harder to distinguish. To address the above challenges, we propose a deep-layer sign-sharing personalized FL (DSPFL) scheme. DSPFL innovatively aggregates only the signs of stochastic gradients (SignSGD) from the deep layers of local models during training. This targeted aggregation enhances the robustness of the shared components against poisoning attacks, while shallow layers are retained locally to preserve personalization. This integrated approach improves the accuracy and resilience of personalized local models on IIoT devices under poisoning attacks. Extensive experimental results show that DSPFL consistently achieves up to 20% higher and more stable overall personalized model accuracy compared to state-of-the-art methods under specific poisoning attacks.

Index Terms -Anomaly detection, federated learning (FL), Internet of Things (IoT), poisoning attack, signs of stochastic gradients (SignSGD).

## I. INTRODUCTION

T HE extensive application of the Industrial Internet of Things (IIoT) around the world prompts the emergence of smart industries [1]. Simultaneously, threats to data security in the smart industry have escalated due to an increasing number of attacks targeting IIoT [2]. Given that sensor data is continuously generated, a promising approach involves developing anomaly detection systems that utilize machine learning (ML) models to identify and classify abnormal data [3]. Conventional ML schemes require sensor data to be collected on a centralized server for training. However, individual organizations often lack su GLYPH&lt;14&gt; cient data to train accurate ML models and are reluctant to share private data, leading to datasilo challenges.

Federated learning (FL) is a distributed ML framework that preserves data privacy and breaks data barriers among participants [4]. In FL, nodes train their ML models locally and upload model parameters (or updates) to an aggregation server. By aggregating these contributions, the server produces an improved global model, which is then returned to the nodes for subsequent training rounds. As IIoT devices often share similar operational goals and capabilities, FL has become a viable solution for training anomaly detection models on IIoTs, thereby enhancing industrial security [5].

However, poisoning attacks hinder the widespread adoption of FL on IIoT, particularly when training datasets across IIoTs are non-independent and identically distributed (non-IID). In poisoning attacks, adversaries attempt to prevent global model convergence or degrade its performance by uploading malicious local model updates. Some poisoned models are easily identifiable by statistical methods, whereas others can be crafted to appear similar to benign ones, making them di GLYPH&lt;14&gt; cult to detect [6]. Furthermore, the prevalent non-IID nature of training datasets across IIoTs typically exacerbates the impact of poisoning attacks. This is because non-IID data naturally cause legitimate local models to di GLYPH&lt;11&gt; er, thereby complicating the statistical analysis (SA) of model parameters and reducing the e GLYPH&lt;11&gt; ectiveness of defenses that rely on detecting outliers. These dual challenges significantly degrade the feasibility of FL for anomaly detection in IIoT environments.

A widely adopted approach to mitigating poisoning attacks involves statistically analyzing model parameters to identify and filter out poisoned local models [7], [8], [9]. However, this method proves ine GLYPH&lt;11&gt; ective when the poisoned local models

See https: // www.ieee.org / publications / rights / index.html for more information.

closely resemble benign ones. Another common strategy is the application of di GLYPH&lt;11&gt; erential privacy (DP) to reduce the influence of poisoned local models [10], [11], [12]. Nevertheless, this technique compromises the utility of the local models, leading to a decline in global model accuracy. Furthermore, these approaches fail to address the personalization needs of nodes with non-IID data distributions.

To address the challenges posed by poisoning attacks while accounting for non-IID data distribution across IIoT devices, this article introduces the deep-layer sign-sharing personalized FL (DSPFL) scheme, designed to enhance the training of anomaly detection models. Specifically, during training, deep neural network (DNN) models on IIoT devices are partitioned into shallow and deep layers. Gradients corresponding to shallow layers are updated locally to facilitate personalized feature extraction, tailored to each device's unique data characteristics. Conversely, updates for deep layers, which typically learn more generalizable, complex patterns, are shared and aggregated across devices.

The core innovation of DSPFL lies in its unique strategy for this aggregation: only the signs of stochastic gradients (SignSGD) from the deep layers are shared and aggregated. This design choice is pivotal for several reasons. By keeping shallow layers personalized, DSPFL allows each node to retain unique characteristics crucial for its local data distribution, making it inherently di GLYPH&lt;14&gt; cult for adversaries to craft universally e GLYPH&lt;11&gt; ective poisoned updates targeting these layers. Furthermore, aggregating only the signs of gradients in the deep layers robustly mitigates the influence of malicious updates. The impact of these malicious updates is diluted by the consensus of honest contributions, as the sign operation curtails the potential for adversaries to disproportionately influence the aggregated update with large-magnitude poisoned gradients. This targeted application of SignSGD to deep layers, responsible for higher-level feature abstraction, o GLYPH&lt;11&gt; ers robust generalization, while the localized shallow layers cater to specific data nuances. Consequently, DSPFL aims to achieve superior personalized model accuracy compared to traditional FL approaches and demonstrates greater resilience to poisoning attacks than other personalized FL methods that do not explicitly integrate such a robust aggregation mechanism.

To validate its e GLYPH&lt;11&gt; ectiveness, extensive experiments were conducted comparing DSPFL to state-of-the-art schemes by training various DNNs (multilayer perceptron (MLP), long short-term memory (LSTM), ResNet-18, Seq2Seq) on diverse datasets (KDD99, CICIDS2017, UNSW-NB15, CIFAR-100, Eng-Fra) under non-IID settings and poisoning attacks. The results demonstrate that DSPFL can achieve up to a 20% improvement in overall personalized model accuracy compared to leading methods under specific poisoning attack scenarios.

The rest of this article is organized as follows. Section II summarizes existing work. In Section III, the research background of personalized FL on IIoT is described. Section IV provides details of the DSPFL scheme. In Section V, the proposed scheme is analyzed from a security perspective. Section VI empirically validates DSPFL. Section VII concludes this article and outlines future work.

## II. RELATED WORK

The related work spans personalized FL, poisoning attacks, SignSGD, and anomaly detection.

## A. Personalized FL

First proposed in 2017 [4], FL is a framework designed to preserve data privacy while enabling e GLYPH&lt;14&gt; cient model updates for participants. A common objective in FL is to train a global model that performs well across all participants' test datasets. However, as data on participants are usually nonIID, significant research e GLYPH&lt;11&gt; orts have focused on developing reliable global models that can cope with data heterogeneity. Solutions include 1) local objective regularization [13], [14], [15], [16]; 2) grouped model aggregation [17]; 3) knowledge distillation [18]; and 4) client selection [19].

Instead of solely optimizing a general global model, some existing work aims to produce personalized models for each participant, i.e., personalized FL (PFL). Several approaches have been proposed, including 1) loss-based local optimization and fine-tuning [20]; 2) multiple local optimization branches [21]; 3) layer-granularity aggregation [22], [23], [24], [25], [26], [27]; 4) model interpolation [28], [29]; 5) model sparsification [28], [30]; and 6) blockchain-based negotiation [31]. Approaches like 1) involving fine-tuning, 2), 4), and 6) can introduce additional computational or communication steps, which might not be ideal for resource-constrained IIoT devices if frequent re-personalization is required. Moreover, tuning personalized sparsification masks in approach 5) can be challenging due to the vast number of parameters in DNNs. Therefore, DSPFL focuses on designing an e GLYPH&lt;11&gt; ective and secure PFL scheme utilizing layer-granularity aggregation, aligning with approach 3), but with a novel integration of robust aggregation.

Despite being a form of layer-granularity aggregation, the proposed DSPFL scheme addresses the drawbacks of existing solutions. For example, HeurpFedLA [25] selects layers with the highest similarity for aggregation, which intuitively benefits global convergence rather than enhancing personalized local accuracy. FedRep [23] keeps only the low-dimensional classifier (head layer) local, limiting its capacity for personalized complex feature extraction from diverse data. FedVF [22], conversely, retains deeper layers locally for personalization. However, without aggregating these deeper layers, local models can be prone to overfitting, necessitating periodic full aggregation of these deep layers, potentially disrupting personalization. Furthermore, the aforementioned schemes often do not explicitly consider security vulnerabilities like poisoning attacks. HeurpFedLA, FedRep, and FedVF are included as benchmarks in Section VI. Two recent PFL schemes, SMCPPFL [32], which uses secure multi-party computation for sub-aggregations, and RobustPFL [33], which employs a layerposition normalized similarity and a blockchain committee, have also been considered for comparison due to their focus on robustness and personalization.

Furthermore, to understand the specific contribution of DSPFL's layer-wise design in integrating sign-based robustness with personalization, we introduce conceptual hybrid

baselines. These involve applying SignSGD globally to the parameter updates generated by established PFL architectures like HeurpFedLA, FedRep, and FedVF, termed SignSGD-HeurpFedLA, SignSGD-FedRep, and SignSGDFedVF, respectively. These allow us to assess if a straightforward combination of SignSGD with existing PFL methods can match DSPFL's performance, or if DSPFL's structural approach of personalizing shallow layers while applying SignSGD only to deep layers o GLYPH&lt;11&gt; ers unique advantages.

## B. Poisoning Attacks

The distributed architecture of FL makes it vulnerable to poisoning attacks. In such attacks, adversaries upload malicious model updates to the parameter server to degrade global model accuracy or prevent convergence [34], [35]. Poisoning attacks can be categorized as model poisoning or data poisoning [36]. Farhadkhani et al. [37] proved the equivalence between model poisoning and data poisoning and derived a practical attack targeting PFL schemes. Pang et al. [38] demonstrated a hybrid approach mixing model and data poisoning for enhanced attack e GLYPH&lt;14&gt; cacy. This article considers both model and data poisoning attacks (DPAs), as well as four latest adaptive model poisoning attacks (MPAs) [39].

Several approaches have been proposed to defend against poisoning attacks, including: 1) SA of model parameters to identify and filter outliers [7], [8], [9]; 2) applying DP by adding noise [10], [11], [12]; and 3) leveraging data augmentation (DA) [40]. However, these methods have limitations: SA can be complex and may fail against sophisticated attacks that mimic benign updates; DP often reduces model utility to achieve privacy / robustness; and DA can be time-intensive. In contrast, DSPFL's design aims for inherent robustness within its aggregation strategy for deep layers, reducing reliance on complex post hoc analyses or global noise addition. For relevance, SA and DP are included as security benchmarks in Section VI.

## C. SignSGD

SignSGD improves communication e GLYPH&lt;14&gt; ciency by transmitting only the signs of local gradients, discarding their magnitudes [41]. Compared to classic stochastic gradient descent (SGD), SignSGD can also o GLYPH&lt;11&gt; er robustness and some privacy benefits, though potentially at the cost of model accuracy if not carefully implemented [12]. To improve SignSGD's model accuracy, existing approaches include 1) introducing accumulated residual errors locally [42] and 2) transmitting an additional scaling factor with the signs [43]. Several FL schemes have incorporated SignSGD to enhance resistance to poisoning attacks [6], [44], [45], [46]. For instance, MVSignSGD [44] uses a majority vote mechanism, and SignSGDFD [46] employs a progressive weighted majority vote. However, these schemes typically prioritize optimizing global model accuracy over personalized performance. This can lead to significant accuracy disparities across individual nodes, particularly in non-IID scenarios, as global sign-based aggregation might suppress locally important gradient information if applied uniformly to all model parameters. DSPFL innovates by applying SignSGD selectively to deep layers while keeping

TABLE I DEEP LEARNING-BASED ANOMALY DETECTION

shallow layers fully personalized, aiming to strike a better balance between personalized model accuracy on non-IID data and robustness against poisoning attacks in FL.

## D. Anomaly Detection

Anomaly detection is the process of identifying data instances that significantly deviate from the majority of data [47]. Several deep learning-based anomaly detection methods have demonstrated superior accuracy compared to traditional techniques. Table I summarizes typical deep learning methods, highlighting their architecture, layer count, and data type. Most research utilizes DNNs with no more than five layers, as also noted in a survey by Pang et al. [47]. Consequently, this article evaluates DSPFL's anomaly detection performance using DNNs with architectures ranging from four to eight layers, and also extends to more complex models like ResNet18 for broader applicability.

Some researchers have adopted PFL for anomaly detection, such as [54] and [55]. Pei et al. [54] fine-tune a multitask neural network for personalized models. Wang et al. [55] balance local and cooperative training by adjusting weight distributions among nodes. However, these schemes do not explicitly address the threat of poisoning attacks, a gap DSPFL aims to fill.

## III. BACKGROUND

IIoT devices typically collect non-IID sensor data due to their diverse operating environments and tasks. For example, for device A, normal temperature data might range from 25 GLYPH&lt;14&gt; C to 30 GLYPH&lt;14&gt; C, whereas for device B, it might range from 125 GLYPH&lt;14&gt; C to 130 GLYPH&lt;14&gt; C. Consequently, what constitutes an anomaly varies: 130 GLYPH&lt;14&gt; C is anomalous for device A, while 30 GLYPH&lt;14&gt; C is anomalous for device B. This non-IID scenario, often termed feature skew [56], necessitates personalized anomaly detection models capable of handling varied data distributions and accurately identifying local anomalies.

Fig. 1. Personalized DNN for anomaly detection in DSPFL. Gradients in shallow layers are updated locally on each IIoT device during training, whereas the signs of gradients in deep layers are shared and aggregated.

<!-- image -->

(via their signs) to collaboratively enhance the models' ability to recognize complex, potentially shared, anomalous patterns. After federated updates to the deep layers, the entire DNN (recombined shallow and deep parts) is trained locally on each IIoT device in the subsequent round. This local training refines the entire model and ensures coherence between the personalized shallow layers and the collaboratively updated deep layers. This process aims to enable each IIoT device's DNN to detect its personalized anomalies accurately without negatively impacting others, while still benefiting from shared insights in recognizing complex patterns.

This PFL strategy also inherently contributes to robustness against poisoning attacks on non-IID data. By confining shallow layer updates locally, any malicious manipulations targeting these device-specific features are not directly propagated to other benign devices. For the shared deep layers, the sign-based aggregation, as detailed later, mitigates the impact of poisoned gradients from adversarial nodes. This dual approach makes DSPFL a compelling design for scenarios demanding both personalization and security.

## IV. MODELING

This section details the proposed DSPFL scheme, covering the threat model, workflow, personalized local training, and deep-layer sign aggregation.

## A. Threat Model

This study considers poisoning attacks as glass-box attacks, where adversaries have compromised some IIoT devices. These adversaries can manipulate (poison) the trained local models or their updates before uploading them to the central server. To comprehensively assess DSPFL's robustness, both model poisoning and DPAs are examined in this article.

- 1) Model Poisoning Attack: Adversaries introduce perturbations (e.g., random noise or strategically crafted updates) to their local model parameters or gradients before uploading them. These modifications aim to degrade the accuracy of the aggregated global components or the final personalized models [59]. By carefully controlling these perturbations, adversaries may attempt to make poisoned updates appear similar to benign ones, evading simple detection mechanisms.
- 2) Data Poisoning Attack: Adversaries inject malicious samples (e.g., mislabeled data) into their local training datasets. Training on such data leads to locally trained models that embed incorrect knowledge, which, if aggregated, can corrupt the shared models [36], [60].

This article assumes that the majority of participants in the IIoT network are honest and contribute benign local model updates. DSPFL's e GLYPH&lt;11&gt; ectiveness against both MPA and DPA under this assumption is evaluated.

## B. Workflow

The process of updating personalized DNNs in DSPFL involves the following four steps in each communication round, as illustrated in Fig. 2.

- 1) Personalized local training: Each IIoT device trains its local DNN model using its local sensor data and anomaly labels. This step updates both shallow and deep layers of the local model.
- 2) Deep-layer sign extracting: After local training, each device computes the change (update) to its deep-layer parameters. It then extracts and uploads only the signs of these deep-layer parameter updates to the parameter server. Shallow-layer updates remain local.
- 3) Deep-layer sign aggregation: The parameter server receives the signs of deep-layer updates from participating devices. It aggregates these signs to determine a consensus update direction for the deep layers.
- 4) Deep-layer parameter update: The server broadcasts the aggregated sign-based update for the deep layers back to all IIoT devices. Each device incorporates this update into its local model's deep layers, recombines them with its personalized shallow layers, and prepares for the next round of local training.

Section IV-C details the personalized local training, and Section IV-D explains the deep-layer sign aggregation.

## C. Personalized Local Training

Fig. 2. Workflow of DSPFL. Local model D is presumed to be poisoned. Its influence on other nodes' deep layers is mitigated by the sign-based gradient aggregation on the parameter server, while its shallow layers remain locally poisoned without direct propagation.

<!-- image -->

Nk is the number of samples for node k . The loss of a local model m k on sample zk ; i is f ( m k ; zk ; i ). The local loss function for node k is

<!-- formula-not-decoded -->

In standard FL, like FedAvg [4], the objective is often to minimize a weighted average of these local losses

<!-- formula-not-decoded -->

where N = P K j = 1 Nj .

In DSPFL, we pursue personalization. The model m k on node k is partitioned into personalized shallow-layer parameters p k and shared (but locally instantiated) deep-layer parameters s k , so m k = ( s k ; p k ). The objective for each node k is to optimize its personalized model by minimizing its local loss Fk ( s k ; p k )

<!-- formula-not-decoded -->

Here, p k is updated strictly locally, while s k is updated based on local training and the aggregated information from the server.

The personalized local training process is detailed in Algorithm 1. Key initializations include the total communication rounds T , learning rate GLYPH&lt;13&gt; , and the initial local model ˆ m 0 k for each node k .

First, the initial local model ˆ m 0 k is split into two parts: shallow layers p 0 k for personalization and deep layers s 0 k for sharing, as shown in Line 1. The GLYPH&lt;8&gt; operator expresses a joint on shallow and deep layers. Specifically, half of the DNN layers near the input are defined as the shallow layers, while the other half of the DNN layers near the output are defined as the deep layers. The parameters (or gradients) of shallow and deep layers are extracted to s 0 k and p 0 k . After that, T rounds of local training start.

After receiving the aggregated deep-layer updates GLYPH&lt;1&gt; s t + 1 from the parameter server, node k applies it to the deep layers

Algorithm 1 Personalized Local Training . Run on IIoT device k

Input: Number of communication rounds T , learning rate GLYPH&lt;13&gt; , initial local model ˆ m 0 k .

- 1: s 0 k GLYPH&lt;8&gt; p 0 k ˆ m 0 k

Output: Personalized local model ˆ m T k .

- 2: for t = 0 ; : : :; T GLYPH&lt;0&gt; 1 do
- 3: m t + 1 k ˆ m t k GLYPH&lt;0&gt; GLYPH&lt;13&gt; ˜ r Fk ( ˆ m t k )
- 5: GLYPH&lt;1&gt; s t + 1 k s t + 1 k GLYPH&lt;0&gt; s t k
- 4: s t + 1 k GLYPH&lt;8&gt; p t + 1 k m t + 1 k
- 6: Upload sgn( GLYPH&lt;1&gt; s t + 1 k ) to the parameter server
- 7: Receive GLYPH&lt;1&gt; s t + 1 from the parameter server
- 8: ˆ s t + 1 k s t k + GLYPH&lt;13&gt; GLYPH&lt;1&gt; s t + 1
- 9: ˆ m t + 1 k ˜ s t + 1 k GLYPH&lt;8&gt; p t + 1 k
- 10: end for

by calculating ˆ s t + 1 k s t k + GLYPH&lt;13&gt; GLYPH&lt;1&gt; s t + 1 , as shown in Lines 7 and 8. Specifically, the learning rate GLYPH&lt;13&gt; is utilized to keep the step size of the optimization consistent with the original local training. Then, the shallow layers are concatenated with the newly generated deep layers, where a new personalized local model ˆ m t + 1 k is generated, as shown in Line 9.

## D. Deep-Layer Sign Aggregation

The parameter server executes the deep-layer sign aggregation process, as outlined in Algorithm 2.

Algorithm 2 Deep-Layer Sign Aggregation . Run on parameter server

Input: Number of communication rounds T .

- 1: for t = 1 ; : : :; T do
- 2: Receive f sgn( GLYPH&lt;1&gt; s t k ) j k 2 K g from IIoT devices
- 3: GLYPH&lt;1&gt; s t sgn GLYPH&lt;16&gt; P K k = 0 sgn( GLYPH&lt;1&gt; s t k ) GLYPH&lt;17&gt;
- 4: Broadcast GLYPH&lt;1&gt; s t to IIoT devices
- 5: end for

After collecting deep-layer sign updates from IIoT devices (Line 2), the server aggregates them. A common method, shown in Line 3, is to sum the sign vectors and then take

Fig. 3. Conceptual illustration of gradient aggregation. Assume five stochastic gradients g 1 to g 5 are collected. g 4 and g 5 are from poisoned models with potentially large magnitudes. Optimal: the ideal optimization direction. FedAvg can be heavily skewed by g 4 ; g 5 . SignSGD (used by DSPFL for deep layers) relies on the majority sign.

<!-- image -->

the sign of the resulting sum. This e GLYPH&lt;11&gt; ectively implements a majority vote for the direction of update for each parameter in the deep layers. This aggregated sign vector GLYPH&lt;1&gt; s t is then broadcast back to the devices.

## V. SECURITY ANALYSIS

DSPFL's robustness to poisoning attacks stems primarily from two aspects: the partial model personalization and the sign-based aggregation for shared layers. By keeping shallow layers local ( p k ), DSPFL confines the direct impact of any poisoning specific to these layers to the compromised client itself. Malicious updates to p k are not propagated. For the shared deep layers ( s k ), sign-based gradient compression significantly curtails the attacker's ability to disproportionately influence the aggregated update. Even if an adversary submits gradients with extremely large magnitudes, their contribution to each coordinate of the sum in Algorithm 2 (Line 3) is capped at ± 1. As long as honest clients form a majority, the sign of the sum (the direction of the aggregated update) is likely to be determined by them.

Fig. 3 illustrates this concept. Gradients from poisoned local models ( g 4 ; g 5) might have large magnitudes and could steer the FedAvg update away from the optimal direction. In contrast, with SignSGD (as used in DSPFL's deep layer aggregation), if the majority of gradients ( g 1 ; g 2 ; g 3) are from honest clients, their consensus on the sign will likely dominate, ensuring the optimization progresses roughly in the correct direction.

Formally, the i th coordinate in g k is denoted as g i k and the optimal one is denoted as ˆ g i . Similar to [44], we assume that g i k obeys unimodal distribution with variance GLYPH&lt;27&gt; i k and the signalto-noise ratio (SNR) for g i k is S i k = ( j g i k j =GLYPH&lt;27&gt; i k ). Based on [41, Lemma D.1], we can get the probability that sgn( g i k ) is not consistent with the optimal coordinate, which is

<!-- formula-not-decoded -->

In particular, the probability in (4) is less than 1 = 2 in all cases. In DSPFL, we also assume that the ratio of adversaries to total nodes GLYPH&lt;11&gt; is always less than 1 = 2, meaning that the majority of the nodes are honest. Therefore, following the proof of [44, Theorem 2], we can get:

<!-- formula-not-decoded -->

Equation (5) illustrates that the probability that the deeplayer updates in Algorithm 2 do not consistent with the optimal direction is bounded. Besides, it can be seen that the probability is decreased when there are fewer adversaries, higher SNR, or more nodes.

Moreover, only the SGDs in deep layers are shared during the training; the SGDs of the shallow layers in the DNN are not a GLYPH&lt;11&gt; ected by sign-based gradient compression or poisoned SGDs at all. As the adversaries may manipulate any parameters in their local models, the probability and impact of poisoning attacks only correlate to the number of parameters in shared deep layers. Let GLYPH&lt;21&gt; represent the ratio of parameters in deep layers to those in all layers. By introducing the function C ( x ) that counts the number of parameters in layer x , we get GLYPH&lt;21&gt; = C ( s ) = [ C ( s ) + C ( p )]. It is easy to know that GLYPH&lt;21&gt; 2 (0 ; 1). Therefore, in terms of the whole DNN, we get

<!-- formula-not-decoded -->

By comparing (6)-(5), it is easy to know that the model-split strategy further reduces the probability that the coordinates in the model are not consistent with the optimal direction. As a result, DSPFL further mitigates the impact of poisoned local models compared to classic SignSGD schemes.

Finally, it is possible to rewrite the non-convex convergence rate of [44, Theorem 2] as

<!-- formula-not-decoded -->

where f 0 represents the objective value at the random starting point, f GLYPH&lt;3&gt; is the lower bound, and L is the Lipschitz constant for L -smooth. The convergence of DSPFL under poisoning attacks is thus guaranteed.

## VI. EVALUATION

This section evaluates the performance of DSPFL regarding personalized model accuracy, resistance to MPAs, and resistance to DPAs. We also analyze the impact of model-split ratios, scalability, communication overhead, and performance on diverse ML tasks.

## A. Experimental Setting

Experiments were conducted on a virtual machine with eight CPU cores and 8 GB of memory. Python 3.8 and PyTorch

v1.8.1 were used on Ubuntu 20.04. By default, N = 10 nodes participated, communication rounds T = 20, and learning rate GLYPH&lt;13&gt; = 0 : 01. All experiments were run three times, and the average results are reported.

To evaluate the performance of personalization, the training and testing datasets across nodes are set to non-IID. Similar to [61], the Zipf distribution is adopted to represent the occurrence frequency of di GLYPH&lt;11&gt; erent classes, while the priorities of classes are randomly picked by each node independently. In particular, the parameter s describes the deviation of the occurrence frequency of classes. Increasing s results in more heterogeneous datasets across nodes. Taking a dataset with ten classes as an example, when s = 0 : 0, the occurrence frequency of all classes on each node is equal, which means that the datasets across nodes are IID. When s = 1 : 5, the occurrence frequency of ten classes becomes 0.50, 0.18, 0.10, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02, and 0.02. As nodes assign classes with di GLYPH&lt;11&gt; erent priorities, the occurrence frequency of a class on a node is 0.5, while it may be 0.03 on another node. As a result, the datasets across nodes are heterogeneous (non-IID). In addition, when s = 2 : 45, the occurrence frequency of ten classes becomes 0.75, 0.14, 0.05, 0.03, 0.01, 0.01, 0.01, 0.0, 0.0, and 0.0. When s = 3 : 70, the occurrence frequency of ten classes is 0.9, 0.07, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, and 0.0.

To evaluate the performance of DSPFL in IIoT security, we employ two foundational DNNs: an MLP and an LSTM network. Using a four-layer configuration as an example, the MLP consists of two linear layers and two ReLU activation layers, while the LSTM comprises four recurrent hidden layers, each with ten hidden units. These architectures are trained on three widely used intrusion detection datasets: KDD99 [62], CICIDS2017 [63], and UNSW-NB15 [64].

The KDD99 dataset contains 19289 samples with four kinds of attacks and is designed to di GLYPH&lt;11&gt; erentiate between attack and normal connections. This dataset simulates network tra GLYPH&lt;14&gt; c and is a foundational benchmark for evaluating intrusion detection models. The CICIDS2017 dataset comprises over 2.8 million samples and eight kinds of attacks. It contains network packets that reflect both benign tra GLYPH&lt;14&gt; c and a wide array of modern attack scenarios, making it ideal for training robust intrusion detection systems. The UNSW-NB15 dataset includes 82332 samples over nine kinds of attacks and features a hybrid of real-world normal activities and synthetic contemporary attack behaviors, providing a more realistic setting for intrusion detection in IIoT environments. These datasets collectively o GLYPH&lt;11&gt; er diverse scenarios and data distributions, simulating IIoT security challenges, where identifying malicious tra GLYPH&lt;14&gt; c and anomalies in network connections is critical.

To further validate the generality of DSPFL beyond intrusion detection tasks, we evaluate its performance on two additional ML tasks: image classification and machine translation. For image classification, a ResNet-18 [65] model is trained on the CIFAR-100 [66] dataset, which consists of 60 000 samples spanning 100 classes. The dataset contains 32 × 32 color images labeled for fine-grained image recognition, representing a challenging task for DNNs due to its high class diversity and compact image resolution. For machine translation, we train a Seq2Seq [67] model on the Eng-Fra dataset, which includes 9701 English-to-French translation pairs. The Seq2Seq architecture features a two-layer LSTM encoder and a two-layer Transformer decoder, following the design outlined in [68]. This task assesses DSPFL's capability to handle sequential and natural language processing (NLP) tasks, further highlighting its adaptability and e GLYPH&lt;11&gt; ectiveness across domains.

Two kinds of poisoning attacks, MPA and DPA, are launched to evaluate the attack resistance performance of different schemes. According to [35], in classic MPA, malicious nodes add noises within a small range [ GLYPH&lt;0&gt; 1 ; 1) to their local models after training, which is challenging to detect. Besides, four adaptive MPAs [39] are included in the experiments: STAT-OPT [35], DNY-OPT, Min-Max, and Min-Sum [9]. STAT-OPT attacks aim to maximize the deviation of global model parameters by pushing them in the inverse direction of their natural update trajectory. DNY-OPT attacks seek to minimize the similarity between compromised and benign global models. The Min-Max attack focuses on maximizing the distance between malicious updates and benign model updates while constraining the maximum distance from other benign updates. Similarly, the Min-Sum approach ensures that the sum of squared distances of malicious gradients from benign updates is bounded by the sum of squared distances between benign updates themselves.

In DPA [36], malicious nodes change the label of the first class to an incorrect (the second) class before training the local models to disrupt the outputs in each training round. The poisoned local models are then treated as normal local models to aggregate in FL. In particular, the accuracy of the personalized model is evaluated against the first class to better elucidate the impact of DPA and assess the DPA resistance of di GLYPH&lt;11&gt; erent schemes.

To evaluate the performance of DSPFL in terms of personalized FL, several state-of-the-art schemes are introduced as benchmarks.

- 1) HeurpFedLA [25]: Select the layers with the highest similarity with other participants to aggregate.
- 2) FedRep [23]: Use local data to learn the personalized head layer of the model.
- 3) FedVF [22]: Aggregate shallow layers more frequently on the server.
- 4) SignSGD [41]: Classic SignSGD scheme.
- 5) FedAvg [4]: Classic FL scheme.
- 6) SignSGD-FD [46]: SignSGD with progressive weighted majority voting mechanism.
- 7) FedProx [13]: Add a proximal term to restrict the local updates to be closer to the global model.
- 8) Sca GLYPH&lt;11&gt; old [14]: Use variance reduction to correct for the client drift in its local updates.

To understand the specific contribution of DSPFL's layerwise design in integrating sign-based robustness with personalization, the following conceptual hybrid baselines are included.

- 1) SignSGD-HeurpFedLA: HeurpFedLA's layer selection and aggregation logic is used, but the aggregated updates are derived from the signs of the selected layer parameters / updates from clients.

TABLE II COMPARISON OF OVERALL PERSONALIZED MODEL ACCURACY (%) ON IID DATASETS (NO ATTACK)

- 2) SignSGD-FedRep: FedRep's structure (personalized head, shared body) is maintained, but the updates for the shared body are aggregated using only their signs.
- 3) SignSGD-FedVF: FedVF's di GLYPH&lt;11&gt; erential aggregation frequency for shallow / deep layers is followed, but all aggregated updates (both shallow and deep, according to FedVF's schedule) are derived from signs.

The aforementioned schemes are not specifically designed to enhance security or resist poisoning attacks. To provide a robust evaluation, several state-of-the-art methods aimed at mitigating poisoning attacks are introduced as benchmarks.

- 1) SA [8]: Outlier rejection.
- 2) DP [11]: Laplace noise addition.
- 3) MV-SignSGD [44]: SignSGD with majority vote.
- 4) SMC-PPFL [32]: Noise perturbation and secure multiparty computation for PFL sub-aggregations.
- 5) RobustPFL [33]: Layer-position normalized similarity and blockchain committee for robust PFL.

In particular, the parameter o in SA represents the number of outliers (malicious local models) that must be excluded prior to aggregation. In the experiments, o is set to 3, corresponding to the three malicious nodes conducting MPA. For DP, the parameter s denotes sensitivity, which determines the exponential decay of the Laplace distribution used for noise addition. Following [69], s is set to 0.2 in the experiments.

## B. Personalized Model Accuracy

To evaluate the performance of DSPFL on IID datasets, the overall personalized model accuracy is compared after 20 rounds of training, as shown in Table II. Specifically, the overall personalized model accuracy is calculated by averaging the local test accuracy of all nodes. When training the MLP model on the KDD99 dataset, DSPFL achieves higher overall personalized model accuracy than other schemes. This reveals that DSPFL has a better capability of knowledge transfer across nodes than HeurpFedLA, FedRep, FedVG, and classic SignSGD. However, the overall personalized model accuracy of DSPFL is lower than that of FedAvg when training the LSTM model on the CICIDS2017 dataset and training the MLP model on the UNSW-NB15 dataset. This

Fig. 4. Overall personalized model accuracy on IID KDD99 dataset (MLP model, no attack).

<!-- image -->

is because the shallow layers of the DNNs are left on nodes for personalized local training, which limits the number of features learned from other nodes compared with FedAvg. Nevertheless, this does not mean that DSPFL is unsatisfactory compared with FedAvg, as DSPFL is originally designed for personalized FL on non-IID datasets. These results only demonstrate that DSPFL has an overall personalized model accuracy that is comparable to the classic FL scheme when facing IID datasets.

Meanwhile, to analyze the convergence speed and the convergence stability of di GLYPH&lt;11&gt; erent schemes, the overall personalized model accuracy is recorded and compared during the training process, as shown in Fig. 4. When training the MLP model on the KDD99 dataset, the convergence speed of DSPFL is comparable to that of HeurpFedLA and FedVF, only one round later than that of FedAvg, SMC-PPFL, and RobustPFL. Besides, SignSGD converges slowly, and FedRep can hardly train a meaningful MLP model. Compared with SignSGD, it is obvious that DSPFL improves the convergence speed due to its model-split strategy and personalized local training.

To evaluate the performance of DSPFL on non-IID datasets, the overall personalized model accuracy is compared during the training process, as shown in Fig. 5. It can be seen that DSPFL always achieves the highest personalized model accuracy and the fastest convergence speed than state-of-the-art schemes. This reveals the e GLYPH&lt;11&gt; ectiveness of personalized local training and the deep-layer sign aggregation algorithms in personalized FL. By contrast, FedAvg and SignSGD have unsatisfactory performance in all circumstances. HeurpFedLA converges stably and fast when training LSTM on CICIDS2017 and training MLP on UNSW-NB15, while it converges slowly when training MLP on KDD99. This is because the aggregation on similarity layers of the MLP model hinders a node from learning more di GLYPH&lt;11&gt; erent features from other nodes. FedRep has unsatisfactory performance when training MLP on KDD99 and training LSTM on CICIDS2017, because the personalization on only the head layer is not enough to extract all features from the KDD99 and CICIDS2017 datasets. FedVF has an unstable convergence during the training, because the less frequent aggregation on deep layers of the DNN ruins the personalization of local models and causes the degradation of personalized model accuracy periodically. SignSGD-FD, while robust under IID settings, fails to maintain competitive personalized model accuracy

Fig. 5. Overall personalized model accuracy on non-IID datasets ( s = 3 : 7, no attack). (a) MLP-KDD99. (b) LSTM-CICIDS2017. (c) MLP-NB15.

<!-- image -->

Fig. 6. Personalized accuracy per node for MLP on non-IID KDD99 ( s = 3 : 7, no attack).

<!-- image -->

under non-IID conditions. Its global optimization strategy sacrifices local node-specific characteristics, which are crucial for personalized FL. Due to the noise perturbation introduced during training, SMC-PPFL converges slightly more slowly than DSPFL and RobustPFL. However, it achieves comparably stable accuracy, attributed to its personalized sub-aggregation mechanism.

To better understand the performance of each local model on its node, the overall personalized model accuracy is evaluated after 20 rounds of training, as shown in Fig. 6. When training MLP on KDD99, DSPFL allows each node to achieve better local test accuracy compared with HeurpFedLA, FedRep, and FedVF, especially for nodes 8 and 9. The reason is that the training datasets on nodes 8 and 9 are largely divergent from the others. For HeurpFedLA, aggregating the layers with the highest similarity ignores the performance of nodes 8 and 9. For FedRep, leaving only the head layer of DNNs reduces the capability of personalized feature extraction. For FedVF, more frequent shallow-layer parameter aggregation results in better generality of feature extraction. As the datasets diverge, the accuracy of models on nodes 8 and 9 is greatly lower than that on the others. By comparison, the personalized local training algorithm in DSPFL e GLYPH&lt;11&gt; ectively increases the accuracy of models on nodes.

To analyze the e GLYPH&lt;11&gt; ect of dataset heterogeneity on model training, the value of parameter s in the Zipf distribution was adjusted from 1.5 to 2.7, as shown in Fig. 7. The occurrence frequencies of di GLYPH&lt;11&gt; erent classes are counted and demonstrated in Fig. 7(a) and (b). It is obvious that class 2 is the dominant class when s = 1 : 5, which has the highest occurrence frequency among all classes and all nodes. As a result, FedAvg has a great performance on nodes 1 and 2, which have training datasets dominated by class 2. Similarly, when s = 2 : 7, class 8 becomes the dominant class and has the

Fig. 7. Top: Example data distribution for UNSW-NB15 (non-IID, s = 1 : 5). Bottom: Corresponding personalized model accuracy of MLP per node. (a) and (b) Dataset distribution s = 1 : 5. (c) s = 1 : 5. (d) s = 2 : 7.

<!-- image -->

highest occurrence frequency on nodes 4, 7, and 8, resulting in FedAvg performing well on nodes 4, 7, and 8, as expected. By contrast, when s = 1 : 5 and s = 2 : 7, the models in DSPFL perform well on all nodes, no matter which class is dominant, due to the capability of personalization. Although the overall personalized model accuracy of DSPFL is decreased from 80% to 50% when the parameter s is decreased from 2.7 to 1.5, the performance of DSPFL is still better than that of FedAvg. This reveals that DSPFL has done its best to allow each node to learn features from other nodes, even when the dataset distribution varies across nodes.

## C. MPA Resistance

To evaluate the e GLYPH&lt;11&gt; ectiveness of DSPFL in resisting MPA, three out of ten nodes continuously launch MPA throughout the training process. The overall personalized model accuracy of various schemes is recorded and compared, as shown in Fig. 8. The results clearly demonstrate that DSPFL is minimally a GLYPH&lt;11&gt; ected by MPA and consistently achieves the highest overall personalized model accuracy among all evaluated schemes. In particular, DSPFL outperforms the second-best one, RobustPFL, by 3% when training an MLP on the KDD99 dataset under MPA, owing to its layerwise and sign-based aggregation strategy. In contrast, state-of-theart schemes either fail to converge or achieve relatively low personalized model accuracy. For example, SA with o = 3 does not perform as expected, as its similarity analysis of models

Fig. 8. Overall personalized model accuracy on non-IID datasets ( s = 3 : 7) with 30% nodes launching MPA. (a) MLP-KDD99. (b) LSTM-CICIDS2017. (c) MLP-NB15.

<!-- image -->

Fig. 9. Personalized accuracy per node for MLP on non-IID KDD99 ( s = 3 : 7) with 30% nodes launching MPA.

<!-- image -->

can be misled when the noise values are su GLYPH&lt;14&gt; ciently small. However, even small noise values can significantly degrade model accuracy, making their impact non-negligible. When employing DP, the personalized model accuracy decreases as the parameter s increases, due to the added noise reducing the utility of the local models. Similar challenges face SMC-PPFL. These findings confirm that DP is not an e GLYPH&lt;11&gt; ective solution for mitigating MPA. MV-SignSGD demonstrates better accuracy and stability compared to FedAvg during the training, indicating that MV-SignSGD o GLYPH&lt;11&gt; ers some resistance to MPA, although there remains considerable room for improvement in terms of personalized model accuracy. FedProx and Sca GLYPH&lt;11&gt; old, on the other hand, show moderate resistance to MPA, but all achieve lower accuracy than DSPFL due to the lack of personalization. In comparison, DSPFL leverages the strengths of both MV-SignSGD and FedAvg, achieving superior results. These findings underscore the e GLYPH&lt;11&gt; ectiveness of DSPFL in maintaining high personalized model accuracy while resisting MPA.

To evaluate the personalized model accuracy of DSPFL under MPA, the accuracy of models on their nodes is recorded after 20 rounds of training, as shown in Fig. 9. When training MLP on KDD99, it is obvious that DSPFL still has a more average performance than FedAvg across nodes, which is similar to that when there are no MPAs. These results validate the e GLYPH&lt;11&gt; ectiveness of DSPFL in both personalized FL and MPA resistance.

To investigate the e GLYPH&lt;11&gt; ect of the number of attackers launching MPA, the overall personalized model accuracy is recorded during the training, as shown in Fig. 10. It is obvious that for DSPFL, the increase in attacker number has some e GLYPH&lt;11&gt; ect on the convergence speed, while it has scarcely any e GLYPH&lt;11&gt; ect on the overall personalized model accuracy. This is in line

Fig. 10. Impact of varying attacker proportions (MPA) on overall personalized accuracy (MLP on non-IID KDD99, s = 3 : 7).

<!-- image -->

TABLE III

COMPARISON OF OVERALL PERSONALIZED MODEL ACCURACY (%) WHEN TRAINING MLP ON KDD99 (NON-IID, s = 3 : 7) UNDER VARIOUS ADAPTIVE MPASS (30% ATTACKERS)

with intuitions, as the majority of the participants are still honest, resulting in the overall optimization of the deep-layer parameters always toward the right direction. By contrast, the performance of FedAvg is significantly a GLYPH&lt;11&gt; ected by the number of attackers, while being worse than that of DSPFL. This is because more nodes poisoning the local models causes the optimizations of the loss functions on nodes toward more divergent directions, which means lower overall personalized model accuracy.

As shown in Table III, DSPFL's e GLYPH&lt;11&gt; ectiveness becomes even more pronounced under sophisticated adaptive poisoning attacks. While competing methods like DP experienced severe accuracy drops to as low as 6.75%, our DSPFL maintained accuracy levels around 57%-60%, demonstrating its robust defense mechanism. This outstanding performance can be attributed to the innovative deep-layer sign-sharing approach, which e GLYPH&lt;11&gt; ectively reduces the impact of attacks aimed at devi-

Fig. 11. Overall personalized model accuracy on the poisoned class for non-IID datasets ( s = 3 : 7) with 30% nodes launching DPA. (a) MLP-KDD99. (b) LSTM-CICIDS2017. (c) MLP-NB15.

<!-- image -->

ating model parameters or manipulating parameter distances, thus providing a more resilient framework for FL in adversarial environments.

## D. DPA Resistance

As three nodes conduct DPA targeting the first class, the personalized model accuracy on the poisoned class is recorded and presented in Fig. 11. When training an MLP on the KDD99 dataset, DSPFL demonstrates superior resistance to DPA, achieving the highest personalized model accuracy on the poisoned class at 47.04%, outperforming SMC-PPFL (32.56%), RobustPFL (42.62%), FedAvg (43.24%), and other schemes. FedProx and Sca GLYPH&lt;11&gt; old perform moderately well but remain less e GLYPH&lt;11&gt; ective than DSPFL. For the LSTM model on the CICIDS2017 dataset, DSPFL achieves a personalized model accuracy of 67.98%, comparable to FedAvg, FedProx, and Sca GLYPH&lt;11&gt; old. This indicates that these schemes are relatively robust against DPA when training an LSTM on CICIDS2017. In contrast, SA (50.00%) and DP (62.57%) fail to ensure convergence, demonstrating their ine GLYPH&lt;11&gt; ectiveness in mitigating DPA. Despite the stable performance of FedAvg, FedProx, and Sca GLYPH&lt;11&gt; old, DSPFL exhibits the fastest convergence speed and the highest stability, further underscoring its robustness. For the MLP model on the UNSW-NB15 dataset, DSPFL achieves a personalized model accuracy of 64.17%, matching the performance of FedRep and surpassing FedProx and Sca GLYPH&lt;11&gt; old by more than 20%. These results indicate that DSPFL provides a higher degree of resilience to DPA compared to FedProx and Sca GLYPH&lt;11&gt; old on this dataset.

## E. Hybrid Baseline Performance

Table IV presents the overall personalized model accuracy of conceptual hybrid baselines compared to their original counterparts and DSPFL on the non-IID KDD99 dataset ( s = 3 : 7) under both no-attack and 30% MPA scenarios. We use the MLP model for this comparison.

As shown in Table IV, applying SignSGD globally to the updates from PFL schemes (SignSGD-HeurpFedLA, SignSGD-FedRep, SignSGD-FedVF) generally improves their robustness against MPA compared to their original counterparts if the original schemes were highly vulnerable (e.g., FedRep's MPA accuracy improves). However, this global application of SignSGD often comes at the cost of reduced personalized accuracy in the no-attack, non-IID scenario. This is because SignSGD, when applied to all shared parameters, can suppress important magnitude information crucial for nuanced

TABLE IV OVERALL PERSONALIZED MODEL ACCURACY (%) ON NON-IID KDD99 (MLP, s = 3 : 7) FOR PFL HYBRIDS WITH SIGNSGD VERSUS DSPFL

personalization that these PFL schemes aim to achieve. For instance, SignSGD-HeurpFedLA and SignSGD-FedVF show lower accuracy than their originals in the no-attack case.

DSPFL outperforms these conceptual SignSGD-PFL hybrids in both no-attack and MPA scenarios on non-IID data. Without attacks, DSPFL's significantly higher accuracy compared to the hybrids demonstrates the superiority of its targeted personalization strategy. Keeping shallow layers entirely local and full-precision allows for better adaptation to non-IID data than attempting to personalize after a global SignSGD aggregation or applying SignSGD over layers that a PFL scheme tries to personalize. When under MPA, DSPFL also maintains a clear advantage over the hybrids. This indicates that DSPFL's strategy of applying SignSGD only to the deep, more generalizable layers provides e GLYPH&lt;11&gt; ective robustness without overly sacrificing the personalization achieved by the local shallow layers. The malicious signals are diluted in the deep layers, while the personalized shallow layers remain una GLYPH&lt;11&gt; ected by direct sign-based compression from other clients. These comparisons suggest that a simple, global application of SignSGD to existing PFL frameworks is not as e GLYPH&lt;11&gt; ective as DSPFL's integrated, layer-aware design.

## F. Impact of Model-Split Ratio

Although in DSPFL the ML models are divided in the middle (i.e., for a four-layer model, there are two shallow layers and two deep layers), it is essential to investigate the performance of other model-split ratios. Therefore, while conducting MPA and DPA, the performance of di GLYPH&lt;11&gt; erent model-split ratios is evaluated and shown in Tables V and VI. In particular, when the model-split ratio is 0:4, no shallow layer parameters are kept locally, and DSPFL is converted to SignSGD. In contrast

Fig. 12. Overall personalized model accuracy on non-IID KDD99 ( s = 3 : 7, no attack) for MLP models with (a) four, (b) six, and (c) eight layers.

<!-- image -->

TABLE V

OVERALL PERSONALIZED MODEL ACCURACY (%) UNDER MPA (30% ATTACKERS) WITH DIFFERENT SHALLOW:DEEP LAYER SPLIT RATIOS

TABLE VI

POISONED CLASS MODEL ACCURACY (%) UNDER DPA (30% ATTACKERS) WITH DIFFERENT SHALLOW:DEEP LAYER SPLIT RATIOS

to DSPFL, the models lose their personalization when dealing with non-IID datasets and become less accurate when facing DPA and MPA. Conversely, when the model-split ratio is 4:0, no deep layer parameters are shared with other nodes, and DSPFL becomes a local ML model training framework. Although it isolates various poisoning attacks toward FL, each node only trains its local model on a small and deviated training dataset due to non-IID settings, resulting in lower personalized model accuracy when compared with DSPFL. Similarly, for model-split ratios of 1:3 and 3:1, it is challenging to strike a balance between personalization, attack resistance, and learnable features. Therefore, among di GLYPH&lt;11&gt; erent modelsplit ratios, separating the anomaly detection models from the middle shows an exceptional performance.

## G. Scalability and Performance on Deeper Models

To further evaluate the scalability and e GLYPH&lt;11&gt; ectiveness of DSPFL, we conducted experiments using both MLPs with varying depths (four, six, and eight layers) on the KDD99 dataset and ResNet-18 on the CIFAR-100 dataset. The results, as shown in Fig. 12, demonstrate that DSPFL consistently outperforms state-of-the-art schemes across all configurations. Specifically, on KDD99, DSPFL achieved superior accuracy for four-layer (65.44%), six-layer (52.18%), and eight-layer (53.24%) MLPs, highlighting its robustness even as model

Fig. 13. Overall personalized model accuracy of DSPFL under MPA (MLP on KDD99, s = 3 : 7) with varying total nodes and attacker ratios.

<!-- image -->

depth increased. Notably, increasing the number of layers did not always result in higher accuracy, likely due to overfitting on the relatively simple KDD99 dataset. Nonetheless, DSPFL maintained the highest accuracy compared to other methods, a GLYPH&lt;14&gt; rming its adaptability to deeper architectures.

To assess the scalability and robustness of DSPFL under MPA across various network sizes, we conducted experiments using the MLP model on the KDD99 dataset with di GLYPH&lt;11&gt; erent numbers of participating nodes and attackers. The results, summarized in Fig. 13, indicate that DSPFL maintains high accuracy across a wide range of network configurations. For smaller networks (e.g., 10 and 20 nodes), DSPFL achieved accuracies of 61.36% and 61.42% with 20% attackers. As the network size increased to 50 and 100 nodes, DSPFL demonstrated consistent or improved performance, achieving 62.73% and 63.21%, respectively. Even with larger attack ratios (e.g., 30% attackers), DSPFL consistently achieved competitive accuracy (62.20%) when there were more nodes (100) in the network. These results illustrate DSPFL's scalability and its ability to e GLYPH&lt;11&gt; ectively mitigate poisoning attacks, even as the network size and complexity increase, making it a robust and adaptable solution for large-scale FL environments.

## H. Performance on Other ML Tasks

To evaluate the generality of DSPFL across di GLYPH&lt;11&gt; erent tasks and model architectures, we conducted experiments on computer vision (CV) and NLP tasks using ResNet-18 on the CIFAR-100 dataset and a Seq2Seq model with an LSTM encoder and Transformer decoder on the Eng-Fra dataset. The experimental results are shown in Fig. 14. For CV tasks, DSPFL achieved an accuracy of 55.1%, outperforming baseline methods including FedAvg (44.1%), HeurpFedLA (48.16%), and FedRep (49.13%). This demonstrates DSPFL's

Fig. 14. Overall personalized model performance of DSPFL versus baselines on CIFAR-100 (ResNet-18, Accuracy, MPA) and Eng-Fra (Seq2Seq, F1-score, MPA). (a) ResNet18 on CIFAR-100. (b) Seq2Seq on Eng-Fra.

<!-- image -->

superior ability to manage complex convolutional neural network (CNN) architectures. For NLP tasks, DSPFL yielded an F1-score of 0.45 on the Eng-Fra dataset, significantly higher than competing methods such as FedAvg (0.27), FedProx (0.38), and Sca GLYPH&lt;11&gt; old (0.36). These results highlight DSPFL's robustness and e GLYPH&lt;11&gt; ectiveness in training both vision and language models, underscoring its generality and scalability across diverse neural network architectures and domains.

## I. Computation and Communication

Local training in DSPFL involves standard backpropagation, comparable to other FL schemes performing local updates. The additional client-side operations (splitting model / gradients, taking signs, reconstructing model) are computationally negligible. Server-side sign aggregation is also lightweight.

To provide a more concrete comparison, Table VII presents per-epoch training time and communication time for various schemes when training an MLP on KDD99 with ten clients. Training time includes local computation on clients and server aggregation. Communication time is the time to transmit model updates from all clients to the server, assuming a representative network bandwidth (e.g., 1 Mb / s uplink per client shared, simplified for illustration). Real-world times will vary based on hardware, network conditions, and implementation details.

DSPFL's training time is among the lowest, comparable to traditional FL methods. This is because its core local training is standard SGD, and the additional steps (splitting, signing) are minimal. Schemes like Sca GLYPH&lt;11&gt; old and SMC-PPFL incur higher computational costs. In addition, DSPFL exhibits the lowest communication time. By transmitting only the signs of half the model parameters (in this example), it significantly reduces the data payload compared to full-precision methods (FedAvg, FedProx, RobustPFL) and even global SignSGD methods (SignSGD, SignSGD-FD). This substantial reduction in communication overhead makes DSPFL particularly attractive for IIoT networks where bandwidth can be limited or costly. While fine-tuning approaches like SMC-PPFL might involve a few epochs, DSPFL integrates personalization directly into its per-round updates, potentially o GLYPH&lt;11&gt; ering a more streamlined and consistently e GLYPH&lt;14&gt; cient process, especially when frequent updates are necessary due to data drift.

## TABLE VII

PER-EPOCH EFFICIENCY COMPARISON IN SECONDS (MLP ON KDD99, TEN NODES). MODEL SIZE: 100K TOTAL PARAMETERS, 32-BIT PRECISION. DSPFL DEEP LAYERS: 50K PARAMETERS. COMMUNICATION: 1 MB/S UPLINK AVAILABLE

## VII. CONCLUSION AND FUTURE WORK

In order to improve the poisoning attack resistance of FL on non-IID datasets, we propose a deep-layer sign-sharing personalized FL scheme. During the training, DNNs are split into shallow and deep layers from the middle. The gradients in shallow layers are kept locally for personalization, while the gradients in deep layers are shared for better complex pattern recognition. Upon that, we innovatively extract and aggregate the signs of the gradients in deep layers, which ensures the accuracy of personalized local models under poisoning attacks. Extensive experiments conducted on non-IID data validate that the proposed scheme e GLYPH&lt;11&gt; ectively improves the overall personalized model accuracy by 20% under certain MPAs. Future work includes exploring adaptive or learned splitting points, identifying malicious local models, and increasing the utility of benign local models.
