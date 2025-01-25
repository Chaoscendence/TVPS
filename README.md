# Beyond MixUps: Task-Guided Variational Partial Synthesis for Fundus Image Analysis
<p align="center">
    <strong>Enyi Li<sup>1,2</sup>, Gaoruishu Long<sup>1,2</sup>, Xuhuan Wang<sup>1,2</sup>, Wei Chen<sup>3</sup>, Jinchao Liu<sup>1,2</sup></strong><br>
    <strong><sup>1</sup>Tianjin Key Laboratory of Intelligent Robotics, College of Artificial Intelligence, Nankai University, China</strong><br>
    <strong><sup>2</sup>Engineering Research Center of Trusted Behavior Intelligence, Ministry of Education, Nankai University, China</strong><br>
    <strong><sup>3</sup>Tianjin Eye Hospital, Nankai University, Tianjin, China</strong><br>

## 📖 Abstract
Though deep learning has shown great potential in the diagnosis and management of ophthalmic diseases, its performance and widespread adoption in ophthalmology are still limited by two key factors, the availability of high-quality training data and in-depth understanding of relations between signals and diseases, especially in the context of few-shot learning. In particular, fundus image analysis, which this study focuses on, has been one of the most cost-effective tools in the daily use of eye clinics, still requires a substantial amount of resources to collect and annotate the images. The resulting datasets are often imbalanced where there are much less samples for rare diseases, which pose a big challenge to image analysis. One promising solution is using generative models which can not only generate virtual samples as a means of data augmentation, but also deepen our understanding of diseases and corresponding signals. However, training generative models with limited data is also non-trivial and requires novel strategies. To this end, we propose a variational generative model, named task-guided variational partial synthesis (TVPS), which generates synthesized local patches online to help train deep models. TVPS is a general VAE-extended probabilistic framework where the neural encoder and decoder are connected via shortcuts and the latent representations learned follow a mixture of Gaussian distribution. Experiments on two public datasets validated the superiority of the proposed method. Moreover, we showed that TVPS is able to generate sensible variations with clinical plausibility for fundus images analysis verified by ophthalmologists.

## 🔥 Logs

- [2025.01.25] The network architectures used in TVPS are available now. And more detailed supplementary information is available, including a detailed derivation of the Evidence Lower Bound (ELBO) and clinical diagnostic opinions from human ophthalmologists on the generated samples.

## 📊 Results

*Here, we provide a brief description of the performance achieved by TVPS.*

**Advanced classification performance:**

We employed two public fundus image datasets to evaluate the proposed TVPS, namely Fundus1000 and Aptos2019. Here is the performance of TVPS on two datasets:

![fundus](./asserts/fundus.png)

![APTOS](./asserts/APTOS.png)

In addition, clinical diagnostic opinions from human ophthalmologists demonstrated that TVPS was able to generate more pronounced clinical features of the disease.

![generated_samples_with_opinion](./asserts/generated_samples_with_opinion.png)
