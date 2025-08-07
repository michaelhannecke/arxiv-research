# An Image is Worth 16x16 Words: Transformers for Image Recognition at   Scale

**Authors**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
**arXiv ID**: 2010.11929
**Published**: 2020-10-22
**Categories**: cs.CV, cs.AI, cs.LG
**Keywords**: self-attention, pre-training, fine-tuning, computational efficiency, scalability, transfer learning, benchmarks (ImageNet, CIFAR-100, VTAB)

## AI Summary
In "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," Dosovitskiy et al. explore the application of Transformer architectures, which have been highly successful in natural language processing, to image recognition tasks. The authors propose the Vision Transformer (ViT), a pure Transformer architecture that can be directly applied to sequences of image patches for image classification, aiming to investigate if Transformers can achieve competitive performance when trained on sufficiently large datasets. The ViT model closely follows the original Transformer design, reshaping input images into sequences of flattened 2D patches and using learnable position embeddings and a classification head attached to the Transformer encoder output. Experiments demonstrate that ViT achieves state-of-the-art performance on various image recognition benchmarks when pre-trained on large datasets, outperforming ResNet-based baselines while requiring substantially less computational resources. The study's findings highlight the potential of Transformer architectures in computer vision, offering a promising alternative to convolutional neural networks and paving the way for further research on the application of Transformers in visual tasks.

## Original Abstract
While the Transformer architecture has become the de-facto standard for
natural language processing tasks, its applications to computer vision remain
limited. In vision, attention is either applied in conjunction with
convolutional networks, or used to replace certain components of convolutional
networks while keeping their overall structure in place. We show that this
reliance on CNNs is not necessary and a pure transformer applied directly to
sequences of image patches can perform very well on image classification tasks.
When pre-trained on large amounts of data and transferred to multiple mid-sized
or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision
Transformer (ViT) attains excellent results compared to state-of-the-art
convolutional networks while requiring substantially fewer computational
resources to train.

## Header

### AI Summary
This is the header section of a research paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The paper was published as a conference paper at ICLR 2021. The authors of the paper are Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby, who are affiliated with the Google Research Brain Team.

### Original Content
PublishedasaconferencepaperatICLR2021
AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
AlexeyDosovitskiy∗,†,LucasBeyer∗,AlexanderKolesnikov∗,DirkWeissenborn∗,
XiaohuaZhai∗,ThomasUnterthiner,MostafaDehghani,MatthiasMinderer,
GeorgHeigold,SylvainGelly,JakobUszkoreit,NeilHoulsby∗,†
∗equaltechnicalcontribution,†equaladvising
GoogleResearch,BrainTeam
{adosovitskiy, neilhoulsby}@google.com

## Abstract

### AI Summary
The Vision Transformer (ViT) is a pure transformer architecture that can be applied directly to sequences of image patches for image classification tasks, without relying on convolutional neural networks (CNNs). When pre-trained on large datasets and transferred to various image recognition benchmarks, ViT achieves excellent results compared to state-of-the-art CNNs while requiring significantly less computational resources for training. This demonstrates that the transformer architecture, which has been successful in natural language processing, can also be effectively applied to computer vision tasks.

### Original Content
WhiletheTransformerarchitecturehasbecomethede-factostandardfornatural
languageprocessingtasks,itsapplicationstocomputervisionremainlimited. In
vision, attentioniseitherappliedinconjunctionwithconvolutionalnetworks, or
usedtoreplacecertaincomponentsofconvolutionalnetworkswhilekeepingtheir
overall structure in place. We show that this reliance on CNNs is not necessary
andapuretransformerapplieddirectlytosequencesofimagepatchescanperform
very well on image classification tasks. When pre-trained on large amounts of
dataandtransferredtomultiplemid-sizedorsmallimagerecognitionbenchmarks
(ImageNet,CIFAR-100,VTAB,etc.),VisionTransformer(ViT)attainsexcellent
results compared to state-of-the-art convolutional networks while requiring sub-
stantiallyfewercomputationalresourcestotrain.1

## Introduction

### AI Summary
This paper explores the application of Transformer architectures, which have been highly successful in natural language processing, to image recognition tasks. While convolutional architectures currently dominate computer vision, the authors aim to investigate if Transformers can achieve competitive performance when trained on sufficiently large datasets, potentially overcoming the lack of inductive biases inherent to CNNs. The paper addresses the gap in existing work by directly applying a standard Transformer to images with minimal modifications and demonstrating that large-scale pre-training can lead to state-of-the-art results on multiple image recognition benchmarks.

### Original Content
Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become
themodelofchoiceinnaturallanguageprocessing(NLP).Thedominantapproachistopre-trainon
alargetextcorpusandthenfine-tuneonasmallertask-specificdataset(Devlinetal.,2019). Thanks
toTransformers’computationalefficiencyandscalability,ithasbecomepossibletotrainmodelsof
unprecedentedsize,withover100Bparameters(Brownetal.,2020;Lepikhinetal.,2020).Withthe
modelsanddatasetsgrowing,thereisstillnosignofsaturatingperformance.
In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989;
Krizhevskyetal.,2012;Heetal.,2016). InspiredbyNLPsuccesses,multipleworkstrycombining
CNN-likearchitectureswithself-attention(Wangetal.,2018;Carionetal.,2020),somereplacing
theconvolutionsentirely(Ramachandranetal.,2019;Wangetal.,2020a). Thelattermodels,while
theoreticallyefficient,havenotyetbeenscaledeffectivelyonmodernhardwareacceleratorsdueto
theuseofspecializedattentionpatterns.Therefore,inlarge-scaleimagerecognition,classicResNet-
likearchitecturesarestillstateoftheart(Mahajanetal.,2018;Xieetal.,2020;Kolesnikovetal.,
2020).
Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard
Transformerdirectlytoimages,withthefewestpossiblemodifications. Todoso,wesplitanimage
intopatchesandprovidethesequenceoflinearembeddingsofthesepatchesasaninputtoaTrans-
former. Imagepatchesaretreatedthesamewayastokens(words)inanNLPapplication. Wetrain
themodelonimageclassificationinsupervisedfashion.
When trained on mid-sized datasets such as ImageNet without strong regularization, these mod-
els yield modest accuracies of a few percentage points below ResNets of comparable size. This
seeminglydiscouragingoutcomemaybeexpected: Transformerslacksomeoftheinductivebiases
1Fine-tuning code and pre-trained models are available at https://github.com/
google-research/vision_transformer
1202
nuJ
]VC.sc[
2v92911.0102:viXra
PublishedasaconferencepaperatICLR2021
inherenttoCNNs,suchastranslationequivarianceandlocality,andthereforedonotgeneralizewell
whentrainedoninsufficientamountsofdata.
However,thepicturechangesifthemodelsaretrainedonlargerdatasets(14M-300Mimages). We
findthatlargescaletrainingtrumpsinductivebias. OurVisionTransformer(ViT)attainsexcellent
results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When
pre-trainedonthepublicImageNet-21kdatasetorthein-houseJFT-300Mdataset,ViTapproaches
or beats state of the art on multiple image recognition benchmarks. In particular, the best model
reachestheaccuracyof88.55%onImageNet,90.72%onImageNet-ReaL,94.55%onCIFAR-100,
and77.63%ontheVTABsuiteof19tasks.

## Related Work

### AI Summary
This section discusses the application of Transformers, originally proposed for machine translation, to image recognition tasks. It highlights various approaches to scale self-attention to images, such as applying attention locally, using sparse approximations, or processing images in patches. The authors also mention the growing interest in combining CNNs with self-attention and the use of large-scale pre-training to make Transformers competitive with state-of-the-art CNNs for image recognition.

### Original Content
Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since be-
come the state of the art method in many NLP tasks. Large Transformer-based models are often
pre-trained on large corpora and then fine-tuned for the task at hand: BERT (Devlin et al., 2019)
usesadenoisingself-supervisedpre-trainingtask,whiletheGPTlineofworkuseslanguagemod-
elingasitspre-trainingtask(Radfordetal.,2018;2019;Brownetal.,2020).
Naive application of self-attention to images would require that each pixel attends to every other
pixel. Withquadraticcostinthenumberofpixels,thisdoesnotscaletorealisticinputsizes. Thus,
toapplyTransformersinthecontextofimageprocessing,severalapproximationshavebeentriedin
thepast. Parmaretal.(2018)appliedtheself-attentiononlyinlocalneighborhoodsforeachquery
pixel instead of globally. Such local multi-head dot-product self attention blocks can completely
replaceconvolutions(Huetal.,2019;Ramachandranetal.,2019;Zhaoetal.,2020). Inadifferent
lineofwork,SparseTransformers(Childetal.,2019)employscalableapproximationstoglobalself-
attentioninordertobeapplicabletoimages. Analternativewaytoscaleattentionistoapplyitin
blocksofvaryingsizes(Weissenbornetal.,2019),intheextremecaseonlyalongindividualaxes(Ho
et al., 2019; Wang et al., 2020a). Many of these specialized attention architectures demonstrate
promising results on computer vision tasks, but require complex engineering to be implemented
efficientlyonhardwareaccelerators.
MostrelatedtooursisthemodelofCordonnieretal.(2020),whichextractspatchesofsize2×2
from the input image and applies full self-attention on top. This model is very similar to ViT,
but our work goes further to demonstrate that large scale pre-training makes vanilla transformers
competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020)
useasmallpatchsizeof2×2pixels, whichmakesthemodelapplicableonlytosmall-resolution
images,whilewehandlemedium-resolutionimagesaswell.
Therehasalsobeenalotofinterestincombiningconvolutionalneuralnetworks(CNNs)withforms
ofself-attention,e.g. byaugmentingfeaturemapsforimageclassification(Belloetal.,2019)orby
furtherprocessingtheoutputofaCNNusingself-attention,e.g.forobjectdetection(Huetal.,2018;
Carionetal.,2020),videoprocessing(Wangetal.,2018;Sunetal.,2019),imageclassification(Wu
etal.,2020),unsupervisedobjectdiscovery(Locatelloetal.,2020),orunifiedtext-visiontasks(Chen
etal.,2020c;Luetal.,2019;Lietal.,2019).
AnotherrecentrelatedmodelisimageGPT(iGPT)(Chenetal.,2020a),whichappliesTransformers
toimagepixelsafterreducingimageresolutionandcolorspace. Themodelistrainedinanunsu-
pervised fashion as a generative model, and the resulting representation can then be fine-tuned or
probedlinearlyforclassificationperformance,achievingamaximalaccuracyof72%onImageNet.
Ourworkaddstotheincreasingcollectionofpapersthatexploreimagerecognitionatlargerscales
thanthestandardImageNetdataset. Theuseofadditionaldatasourcesallowstoachievestate-of-
the-artresultsonstandardbenchmarks(Mahajanetal.,2018;Touvronetal.,2019;Xieetal.,2020).
Moreover,Sunetal.(2017)studyhowCNNperformancescaleswithdatasetsize,andKolesnikov
etal.(2020);Djolongaetal.(2020)performanempiricalexplorationofCNNtransferlearningfrom
largescaledatasetssuchasImageNet-21kandJFT-300M.Wefocusonthesetwolatterdatasetsas
well,buttrainTransformersinsteadofResNet-basedmodelsusedinpriorworks.
PublishedasaconferencepaperatICLR2021
Vision Transformer (ViT) Transformer Encoder
Class L x +
Bird MLP
Ball Head
Car MLP
Norm
Transformer Encoder
Patch + Position 0 1 2 3 4 5 6 7 8 9 Multi-Head Embedding Attention
* Extra learnable
[class] embedding Linear Projection of Flattened Patches
Norm
Embedded
Patches
Figure1: Modeloverview. Wesplitanimageintofixed-sizepatches,linearlyembedeachofthem,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder.Inordertoperformclassification,weusethestandardapproachofaddinganextralearnable
“classificationtoken”tothesequence. TheillustrationoftheTransformerencoderwasinspiredby
Vaswanietal.(2017).

## Method

### AI Summary
The authors closely follow the design of the original Transformer architecture from Vaswani et al. (2017) for their image recognition model. This simple setup allows them to leverage existing scalable NLP Transformer architectures and their efficient implementations with minimal modifications. By adapting the Transformer architecture, which has been successful in NLP tasks, to image recognition, the authors aim to improve upon existing approaches in terms of scalability and performance.

### Original Content
In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible.
AnadvantageofthisintentionallysimplesetupisthatscalableNLPTransformerarchitectures–and
theirefficientimplementations–canbeusedalmostoutofthebox.

## 3.1 Visiontransformer(Vit)

### AI Summary
The Vision Transformer (ViT) model reshapes an input image into a sequence of flattened 2D patches, which are linearly projected to a constant latent vector size and serve as input to a standard Transformer encoder. The model uses learnable position embeddings and a classification head attached to the output of the Transformer encoder. Compared to CNNs, ViT has less image-specific inductive bias, with self-attention layers being global and only MLP layers being local and translationally equivariant.

### Original Content
AnoverviewofthemodelisdepictedinFigure1. ThestandardTransformerreceivesasinputa1D
sequenceoftokenembeddings. Tohandle2Dimages,wereshapetheimagex ∈ RH×W×C intoa
sequenceofflattened2Dpatchesx ∈ RN×(P2·C),where(H,W)istheresolutionoftheoriginal
image,Cisthenumberofchannels,(P,P)istheresolutionofeachimagepatch,andN =HW/P2
istheresultingnumberofpatches,whichalsoservesastheeffectiveinputsequencelengthforthe
Transformer. The Transformer uses constant latent vector size D through all of its layers, so we
flattenthepatchesandmaptoD dimensionswithatrainablelinearprojection(Eq.1). Wereferto
theoutputofthisprojectionasthepatchembeddings.
SimilartoBERT’s[class]token,weprependalearnableembeddingtothesequenceofembed-
dedpatches(z0 = x ), whosestateattheoutputoftheTransformerencoder(z0)servesasthe
0 class L
imagerepresentationy(Eq.4). Bothduringpre-trainingandfine-tuning,aclassificationheadisat-
tachedtoz0.TheclassificationheadisimplementedbyaMLPwithonehiddenlayeratpre-training
timeandbyasinglelinearlayeratfine-tuningtime.
Position embeddings are added to the patch embeddings to retain positional information. We use
standard learnable 1D position embeddings, since we have not observed significant performance
gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting
sequenceofembeddingvectorsservesasinputtotheencoder.
TheTransformerencoder(Vaswanietal.,2017)consistsofalternatinglayersofmultiheadedself-
attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
everyblock,andresidualconnectionsaftereveryblock(Wangetal.,2019;Baevski&Auli,2019).
PublishedasaconferencepaperatICLR2021
TheMLPcontainstwolayerswithaGELUnon-linearity.
z =[x ; x1E; x2E;··· ; xNE]+E , E∈R(P2·C)×D, E ∈R(N+1)×D (1)
0 class p p p pos pos
z(cid:48) =MSA(LN(z ))+z , (cid:96)=1...L (2)
(cid:96) (cid:96)−1 (cid:96)−1
z =MLP(LN(z(cid:48) ))+z(cid:48) , (cid:96)=1...L (3)
(cid:96) (cid:96) (cid:96)
y=LN(z0) (4)
Inductivebias. WenotethatVisionTransformerhasmuchlessimage-specificinductivebiasthan
CNNs.InCNNs,locality,two-dimensionalneighborhoodstructure,andtranslationequivarianceare
bakedintoeachlayerthroughoutthewholemodel. InViT,onlyMLPlayersarelocalandtransla-
tionallyequivariant, whiletheself-attentionlayersareglobal. Thetwo-dimensionalneighborhood
structureisusedverysparingly:inthebeginningofthemodelbycuttingtheimageintopatchesand
atfine-tuningtimeforadjustingthepositionembeddingsforimagesofdifferentresolution(asde-
scribedbelow). Otherthanthat,thepositionembeddingsatinitializationtimecarrynoinformation
aboutthe2Dpositionsofthepatchesandallspatialrelationsbetweenthepatcheshavetobelearned
fromscratch.
HybridArchitecture. Asanalternativetorawimagepatches,theinputsequencecanbeformed
from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding
projection E (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case,
the patches can have spatial size 1x1, which means that the input sequence is obtained by simply
flattening the spatial dimensions of the feature map and projecting to the Transformer dimension.
Theclassificationinputembeddingandpositionembeddingsareaddedasdescribedabove.

## 3.2 Fine-Tuningandhigherresolution

### AI Summary
When fine-tuning a pre-trained Vision Transformer (ViT) model on a downstream task, it is often beneficial to use higher resolution images than those used during pre-training. To accommodate the increased resolution, the patch size is kept constant, resulting in a larger sequence length. The pre-trained position embeddings are interpolated in 2D to match the new sequence length, which is the only point where an inductive bias about the 2D structure of the images is manually introduced into the ViT model.

### Original Content
Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For
this, we remove the pre-trained prediction head and attach a zero-initialized D ×K feedforward
layer, where K is the number of downstream classes. It is often beneficial to fine-tune at higher
resolution than pre-training (Touvron et al.,2019; Kolesnikov et al., 2020). When feeding images
of higher resolution, we keep the patch size the same, which results in a larger effective sequence
length. TheVisionTransformercanhandlearbitrarysequencelengths(uptomemoryconstraints),
however,thepre-trainedpositionembeddingsmaynolongerbemeaningful. Wethereforeperform
2Dinterpolationofthepre-trainedpositionembeddings,accordingtotheirlocationintheoriginal
image. Note that this resolution adjustment and patch extraction are the only points at which an
inductivebiasaboutthe2DstructureoftheimagesismanuallyinjectedintotheVisionTransformer.

## Experiments

### AI Summary
The experiments evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and their hybrid on datasets of varying sizes, considering the computational cost of pre-training. ViT achieves state-of-the-art performance on most recognition benchmarks at a lower pre-training cost compared to baselines. Additionally, a small experiment using self-supervision shows promising results for the future of self-supervised ViT models.

### Original Content
We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the
hybrid. Tounderstandthedatarequirementsofeachmodel,wepre-trainondatasetsofvaryingsize
andevaluatemanybenchmarktasks. Whenconsideringthecomputationalcostofpre-trainingthe
model, ViTperformsveryfavourably, attainingstateoftheartonmostrecognitionbenchmarksat
alowerpre-trainingcost. Lastly, weperformasmallexperimentusingself-supervision, andshow
thatself-supervisedViTholdspromiseforthefuture.

## 4.1 Setup

### AI Summary
This section describes the setup for experiments comparing Vision Transformer (ViT) models to CNNs. The models are trained on large datasets like ImageNet, ImageNet-21k, and JFT, then evaluated on several benchmark tasks including CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102, and the 19-task VTAB suite. Details are provided on the ViT model variants, modifications made to the baseline ResNet models, and the training procedures used.

### Original Content
Datasets. Toexploremodelscalability,weusetheILSVRC-2012ImageNetdatasetwith1kclasses
and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with
21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and
303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the
downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these
dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up
ReaLlabels(Beyeretal.,2020),CIFAR-10/100(Krizhevsky,2009),Oxford-IIITPets(Parkhietal.,
2012), andOxfordFlowers-102(Nilsback&Zisserman,2008). Forthesedatasets, pre-processing
followsKolesnikovetal.(2020).
PublishedasaconferencepaperatICLR2021
Model Layers HiddensizeD MLPsize Heads Params
ViT-Base 12 768 3072 12 86M
ViT-Large 24 1024 4096 16 307M
ViT-Huge 32 1280 5120 16 632M
Table1: DetailsofVisionTransformermodelvariants.
We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates
low-datatransfertodiversetasks,using1000trainingexamplespertask. Thetasksaredividedinto
threegroups: Natural–tasksliketheabove,Pets,CIFAR,etc. Specialized –medicalandsatellite
imagery,andStructured–tasksthatrequiregeometricunderstandinglikelocalization.
Model Variants. We base ViT configurations on those used for BERT (Devlin et al., 2019), as
summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we
addthelarger“Huge”model. Inwhatfollowsweusebriefnotationtoindicatethemodelsizeand
theinputpatchsize:forinstance,ViT-L/16meansthe“Large”variantwith16×16inputpatchsize.
NotethattheTransformer’ssequencelengthisinverselyproportionaltothesquareofthepatchsize,
thusmodelswithsmallerpatchsizearecomputationallymoreexpensive.
ForthebaselineCNNs,weuseResNet(Heetal.,2016),butreplacetheBatchNormalizationlay-
ers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized
convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020),
andwedenotethemodifiedmodel“ResNet(BiT)”.Forthehybrids, wefeedtheintermediatefea-
turemapsintoViTwithpatchsizeofone“pixel”. Toexperimentwithdifferentsequencelengths,
weeither(i)taketheoutputofstage4ofaregularResNet50or(ii)removestage4,placethesame
numberoflayersinstage3(keepingthetotalnumberoflayers),andtaketheoutputofthisextended
stage3. Option(ii)resultsina4xlongersequencelength,andamoreexpensiveViTmodel.
Training & Fine-tuning. We train all models, including ResNets, using Adam (Kingma & Ba,
2015)withβ =0.9,β =0.999,abatchsizeof4096andapplyahighweightdecayof0.1,which
1 2
wefoundtobeusefulfortransferofallmodels(AppendixD.1showsthat,incontrasttocommon
practices,AdamworksslightlybetterthanSGDforResNetsinoursetting).Weusealinearlearning
ratewarmupanddecay,seeAppendixB.1fordetails. Forfine-tuningweuseSGDwithmomentum,
batchsize512,forallmodels,seeAppendixB.1.1.ForImageNetresultsinTable2,wefine-tunedat
higherresolution: 512forViT-L/16and518forViT-H/14,andalsousedPolyak&Juditsky(1992)
averagingwithafactorof0.9999(Ramachandranetal.,2019;Wangetal.,2020b).
Metrics. Wereportresultsondownstreamdatasetseitherthroughfew-shotorfine-tuningaccuracy.
Fine-tuningaccuraciescapturetheperformanceofeachmodelafterfine-tuningitontherespective
dataset. Few-shotaccuraciesareobtainedbysolvingaregularizedleast-squaresregressionproblem
thatmapsthe(frozen)representationofasubsetoftrainingimagesto{−1,1}K targetvectors. This
formulation allows us to recover the exact solution in closed form. Though we mainly focus on
fine-tuningperformance,wesometimesuselinearfew-shotaccuraciesforfaston-the-flyevaluation
wherefine-tuningwouldbetoocostly.

## 4.2 Comparisontostateoftheart

### AI Summary
This section compares the performance of Vision Transformer (ViT) models to state-of-the-art CNNs on various image classification benchmarks. The ViT models, pre-trained on the large JFT-300M dataset, outperform ResNet-based baselines on all datasets while requiring substantially less computational resources for pre-training. Additionally, the ViT model pre-trained on the smaller public ImageNet-21k dataset also performs well on most datasets and can be trained using standard hardware in a reasonable amount of time.

### Original Content
We first compare our largest models – ViT-H/14 and ViT-L/16 – to state-of-the-art CNNs from
the literature. The first comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which
performssupervisedtransferlearningwithlargeResNets. ThesecondisNoisyStudent(Xieetal.,
2020), whichisalargeEfficientNettrainedusingsemi-supervisedlearningonImageNetandJFT-
300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and
BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we
reportthenumberofTPUv3-core-daystakentopre-traineachofthem,thatis,thenumberofTPU
v3cores(2perchip)usedfortrainingmultipliedbythetrainingtimeindays.
Table2showstheresults.ThesmallerViT-L/16modelpre-trainedonJFT-300MoutperformsBiT-L
(whichispre-trainedonthesamedataset)onalltasks,whilerequiringsubstantiallylesscomputa-
tionalresourcestotrain. Thelargermodel,ViT-H/14,furtherimprovestheperformance,especially
onthemorechallengingdatasets–ImageNet,CIFAR-100,andtheVTABsuite. Interestingly,this
PublishedasaconferencepaperatICLR2021
Ours-JFT Ours-JFT Ours-I21k BiT-L NoisyStudent
(ViT-H/14) (ViT-L/16) (ViT-L/16) (ResNet152x4) (EfficientNet-L2)
ImageNet 88.55±0.04 87.76±0.03 85.30±0.02 87.54±0.02 88.4/88.5∗
ImageNetReaL 90.72±0.05 90.54±0.03 88.62±0.05 90.54 90.55
CIFAR-10 99.50±0.06 99.42±0.03 99.15±0.03 99.37±0.06 −
CIFAR-100 94.55±0.04 93.90±0.05 93.25±0.05 93.51±0.08 −
Oxford-IIITPets 97.56±0.03 97.32±0.11 94.67±0.15 96.62±0.23 −
OxfordFlowers-102 99.68±0.02 99.74±0.00 99.61±0.02 99.63±0.03 −
VTAB(19tasks) 77.63±0.23 76.28±0.46 72.72±0.21 76.29±1.70 −
TPUv3-core-days 2.5k 0.68k 0.23k 9.9k 12.3k
Table 2: Comparison with state of the art on popular image classification benchmarks. We re-
port mean and standard deviation of the accuracies, averaged over three fine-tuning runs. Vision
Transformermodelspre-trainedontheJFT-300MdatasetoutperformResNet-basedbaselinesonall
datasets,whiletakingsubstantiallylesscomputationalresourcestopre-train. ViTpre-trainedonthe
smallerpublicImageNet-21kdatasetperformswelltoo. ∗Slightlyimproved88.5%resultreported
inTouvronetal.(2020).
80 90 90 70
ViT-H/14 BiT-L (R152x4) VIVI-Ex-100% (R50x3) S4L (R50x1)
80 85 60
65 70 80 50
VTAB (19 tasks) Natural (7 tasks) Specialized (4 tasks) Structured (8 tasks)
Figure2: BreakdownofVTABperformanceinNatural,Specialized,andStructuredtaskgroups.
modelstilltooksubstantiallylesscomputetopre-trainthanpriorstateoftheart. However,wenote
that pre-training efficiency may be affected not only by the architecture choice, but also other pa-
rameters,suchastrainingschedule,optimizer,weightdecay,etc. Weprovideacontrolledstudyof
performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model
pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking
fewerresourcestopre-train: itcouldbetrainedusingastandardcloudTPUv3with8coresinap-
proximately30days.
Figure2decomposestheVTABtasksintotheirrespectivegroups,andcomparestopreviousSOTA
methodsonthisbenchmark:BiT,VIVI–aResNetco-trainedonImageNetandYoutube(Tschannen
etal.,2020),andS4L–supervisedplussemi-supervisedlearningonImageNet(Zhaietal.,2019a).
ViT-H/14outperformsBiT-R152x4,andothermethods,ontheNaturalandStructuredtasks.Onthe
Specializedtheperformanceofthetoptwomodelsissimilar.

## 4.3 Pre-Trainingdatarequirements

### AI Summary
This section investigates the impact of pre-training dataset size on the performance of Vision Transformers (ViT) compared to ResNets. Experiments show that while ViT models underperform ResNets when pre-trained on smaller datasets like ImageNet, they outperform ResNets when pre-trained on larger datasets such as ImageNet-21k and JFT-300M. The results also demonstrate that larger ViT variants benefit more from increased pre-training dataset sizes compared to smaller ViT models.

### Original Content
TheVisionTransformerperformswellwhenpre-trainedonalargeJFT-300Mdataset. Withfewer
inductivebiasesforvisionthanResNets,howcrucialisthedatasetsize? Weperformtwoseriesof
experiments.
First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-
300M. To boost the performance on the smaller datasets, we optimize three basic regularization
parameters – weight decay, dropout, and label smoothing. Figure 3 shows the results after fine-
tuning to ImageNet (results on other datasets are shown in Table 5)2. When pre-trained on the
smallestdataset,ImageNet,ViT-LargemodelsunderperformcomparedtoViT-Basemodels,despite
(moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only
with JFT-300M, do we see the full benefit of larger models. Figure 3 also shows the performance
2NotethattheImageNetpre-trainedmodelsarealsofine-tuned,butagainonImageNet.Thisisbecausethe
resolutionincreaseduringfine-tuningimprovestheperformance.
ycaruccA
PublishedasaconferencepaperatICLR2021
75 BiT ViT-L/32 40
ViT-B/32 ViT-L/16 ViT-L/16 ViT-B/32 ResNet50x1 (BiT)
ViT-B/16 ViT-H/14 ViT-L/32 ViT-b/32 ResNet152x2 (BiT)
70 30
ImageNet ImageNet-21k JFT-300M 10 M 30 M 100 M 300 M
Pre-training dataset Number of JFT pre-training samples
Figure 3: Transfer to ImageNet. While Figure 4: Linear few-shot evaluation on Ima-
large ViT models perform worse than BiT geNet versus pre-training size. ResNets per-
ResNets (shaded area) when pre-trained on form better with smaller pre-training datasets
smalldatasets,theyshinewhenpre-trainedon but plateau sooner than ViT, which performs
larger datasets. Similarly, larger ViT variants betterwithlargerpre-training. ViT-bisViT-B
overtakesmalleronesasthedatasetgrows. withallhiddendimensionshalved.
Average-5 ImageNet
Transformer (ViT) Transformer (ViT)
ResNet (BiT) ResNet (BiT)
Hybrid Hybrid
102 103 102 103
Total pre-training compute [exaFLOPs]
Figure5:Performanceversuspre-trainingcomputefordifferentarchitectures:VisionTransformers,
ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same compu-
tational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap
vanishesforlargermodels.
regionspannedbyBiTmodelsofdifferentsizes. TheBiTCNNsoutperformViTonImageNet,but
withthelargerdatasets,ViTovertakes.
Second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT-
300Mdataset. Wedonotperformadditionalregularizationonthesmallersubsetsandusethesame
hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the
effectofregularization. Wedo,however,useearly-stopping,andreportthebestvalidationaccuracy
achievedduringtraining. Tosavecompute,wereportfew-shotlinearaccuracyinsteadoffullfine-
tuningaccuracy. Figure4containstheresults. VisionTransformersoverfitmorethanResNetswith
comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than
ResNet50;itperformsmuchworseonthe9Msubset,butbetteron90M+subsets. Thesameistrue
forResNet152x2andViT-L/16. Thisresultreinforcestheintuitionthattheconvolutionalinductive
bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from
dataissufficient,evenbeneficial.
Overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB
(Table2)seempromisingforverylow-datatransfer. Furtheranalysisoffew-shotpropertiesofViT
isanexcitingdirectionoffuturework.
ycaruccA
1poT
teNegamI
ycarucca
refsnarT
1poT
teNegamI
tohs-5
raeniL
PublishedasaconferencepaperatICLR2021

## 4.4 Scalingstudy

### AI Summary
This section compares the performance and computational cost of different computer vision models, including ResNets, Vision Transformers (ViT), and hybrid models, when trained on a large dataset (JFT-300M). The study finds that Vision Transformers outperform ResNets in terms of the performance-to-compute ratio, requiring 2-4 times less computation to achieve the same performance. Additionally, the results suggest that Vision Transformers have the potential for further improvement with increased scale, while hybrid models only slightly outperform ViT at lower computational budgets.

### Original Content
Weperformacontrolledscalingstudyofdifferentmodelsbyevaluatingtransferperformancefrom
JFT-300M. In this setting data size does not bottleneck the models’ performances, and we assess
performance versus pre-training cost of each model. The model set includes: 7 ResNets, R50x1,
R50x2 R101x1, R152x1, R152x2, pre-trained for 7 epochs, plus R152x2 and R200x3 pre-trained
for 14 epochs; 6 Vision Transformers, ViT-B/32, B/16, L/32, L/16, pre-trained for 7 epochs, plus
L/16 and H/14 pre-trained for 14 epochs; and 5 hybrids, R50+ViT-B/32, B/16, L/32, L/16 pre-
trainedfor7epochs,plusR50+ViT-L/16pre-trainedfor14epochs(forhybrids,thenumberatthe
endofthemodelnamestandsnotforthepatchsize,butforthetotaldowsamplingratiointheResNet
backbone).
Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.5
for details on computational costs). Detailed results per model are provided in Table 6 in the Ap-
pendix. A few patterns can be observed. First, Vision Transformers dominate ResNets on the
performance/compute trade-off. ViT uses approximately 2−4× less compute to attain the same
performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small compu-
tationalbudgets,butthedifferencevanishesforlargermodels. Thisresultissomewhatsurprising,
sinceonemightexpectconvolutionallocalfeatureprocessingtoassistViTatanysize.Third,Vision
Transformersappearnottosaturatewithintherangetried,motivatingfuturescalingefforts.

## 4.5 Inspectingvisiontransformer

### AI Summary
The Vision Transformer learns meaningful patch embeddings and position embeddings that capture image structure and topology. Through self-attention, the model integrates information across the entire image, with some attention heads attending globally even in the lowest layers, while others maintain localized attention, potentially serving a similar function as early convolutional layers in CNNs. The attention distance increases with network depth, and the model focuses on semantically relevant regions for classification.

### Original Content
Tobegintounderstandhow theVisionTransformerprocessesim-
agedata,weanalyzeitsinternalrepresentations. Thefirstlayerof Input Attention
theVisionTransformerlinearlyprojectstheflattenedpatchesintoa
lower-dimensionalspace(Eq.1). Figure7(left)showsthetopprin-
cipal components of the the learned embedding filters. The com-
ponents resemble plausible basis functions for a low-dimensional
representationofthefinestructurewithineachpatch.
After the projection, a learned position embedding is added to the
patchrepresentations.Figure7(center)showsthatthemodellearns
toencodedistancewithintheimageinthesimilarityofpositionem-
beddings,i.e. closerpatchestendtohavemoresimilarpositionem-
beddings. Further,therow-columnstructureappears;patchesinthe
same row/column have similar embeddings. Finally, a sinusoidal
structureissometimesapparentforlargergrids(AppendixD).That
thepositionembeddingslearntorepresent2Dimagetopologyex-
plainswhyhand-crafted2D-awareembeddingvariantsdonotyield
improvements(AppendixD.4).
Self-attentionallowsViTtointegrateinformationacrosstheentire Figure 6: Representative ex-
image even in the lowest layers. We investigate to what degree amples of attention from the
thenetworkmakesuseofthiscapability. Specifically,wecompute output token to the input
the average distance in image space across which information is space. See Appendix D.7 for
integrated, based on the attention weights (Figure 7, right). This details.
“attention distance” is analogous to receptive field size in CNNs.
We find that some heads attend to most of the image already in the lowest layers, showing that
the ability to integrate information globally is indeed used by the model. Other attention heads
have consistently small attention distances in the low layers. This highly localized attention is
less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right),
suggestingthatitmayserveasimilarfunctionasearlyconvolutionallayersinCNNs. Further,the
attentiondistanceincreaseswithnetworkdepth. Globally,wefindthatthemodelattendstoimage
regionsthataresemanticallyrelevantforclassification(Figure6).

## 4.6 Self-Supervision

### AI Summary
This section explores the impact of self-supervised pre-training on the performance of Vision Transformers (ViT) for image recognition tasks. By employing masked patch prediction, mimicking the masked language modeling used in BERT, the smaller ViT-B/16 model achieves a significant improvement of 2% in accuracy on ImageNet compared to training from scratch. However, this self-supervised pre-training approach still falls 4% behind supervised pre-training, and the authors suggest exploring contrastive pre-training in future work.

### Original Content
Transformers show impressive performance on NLP tasks. However, much of their success stems
notonlyfromtheirexcellentscalabilitybutalsofromlargescaleself-supervisedpre-training(Devlin
PublishedasaconferencepaperatICLR2021
RGB embedding filters
(first 28 principal components)
Position embedding similarity ViT-L/16
1 120
2 100
3 80
4 60 5 40 H He ea ad d 1 2
6 20 Head 3
1 0
1 2 3 4 5 6 7 0 5 10 15 20
Input patch column Network depth (layer)
Figure7: Left: FiltersoftheinitiallinearembeddingofRGBvaluesofViT-L/32. Center: Sim-
ilarity of position embeddings of ViT-L/32. Tiles show the cosine similarity between the position
embeddingofthepatchwiththeindicatedrowandcolumnandthepositionembeddingsofallother
patches.Right:Sizeofattendedareabyheadandnetworkdepth.Eachdotshowsthemeanattention
distanceacrossimagesforoneof16headsatonelayer. SeeAppendixD.7fordetails.
et al., 2019; Radford et al., 2018). We also perform a preliminary exploration on masked patch
predictionforself-supervision,mimickingthemaskedlanguagemodelingtaskusedinBERT.With
self-supervisedpre-training,oursmallerViT-B/16modelachieves79.9%accuracyonImageNet,a
significantimprovementof2%totrainingfromscratch,butstill4%behindsupervisedpre-training.
Appendix B.1.2 contains further details. We leave exploration of contrastive pre-training (Chen
etal.,2020b;Heetal.,2020;Bachmanetal.,2019;He´naffetal.,2020)tofuturework.

## Conclusion

### AI Summary
This paper explores the direct application of Transformers to image recognition, treating an image as a sequence of patches processed by a standard Transformer encoder. The results show that this simple yet scalable strategy works well when coupled with pre-training on large datasets, matching or exceeding state-of-the-art performance on many image classification tasks. Future research directions include applying Vision Transformers to other computer vision tasks, exploring self-supervised pre-training methods, and further scaling of the model for improved performance.

### Original Content
WehaveexploredthedirectapplicationofTransformerstoimagerecognition. Unlikepriorworks
using self-attention in computer vision, we do not introduce image-specific inductive biases into
the architecture apart from the initial patch extraction step. Instead, we interpret an image as a
sequenceofpatchesandprocessitbyastandardTransformerencoderasusedinNLP.Thissimple,
yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets.
Thus, Vision Transformer matches or exceeds the state of the art on many image classification
datasets,whilstbeingrelativelycheaptopre-train.
While these initial results are encouraging, many challenges remain. One is to apply ViT to other
computervisiontasks,suchasdetectionandsegmentation.Ourresults,coupledwiththoseinCarion
etal.(2020),indicatethepromiseofthisapproach. Anotherchallengeistocontinueexploringself-
supervised pre-training methods. Our initial experiments show improvement from self-supervised
pre-training, but there is still large gap between self-supervised and large-scale supervised pre-
training. Finally,furtherscalingofViTwouldlikelyleadtoimprovedperformance.

## Acknowledgements

### AI Summary
The research for this paper was conducted collaboratively across Google offices in Berlin, Zurich, and Amsterdam. The authors express gratitude to several Google colleagues for their valuable contributions, including Andreas Steiner for infrastructure support and open-source code release, Joan Puigcerver and Maxim Neumann for assistance with large-scale training infrastructure, and Dmitry Lepikhin, Aravindh Mahendran, Daniel Keysers, Mario Lučić, Noam Shazeer, Ashish Vaswani, and Colin Raffel for insightful discussions that helped shape the research.

### Original Content
TheworkwasperformedinBerlin,Zu¨rich,andAmsterdam. WethankmanycolleaguesatGoogle
for their help, in particular Andreas Steiner for crucial help with the infrastructure and the open-
source release of the code; Joan Puigcerver and Maxim Neumann for help with the large-scale
traininginfrastructure;DmitryLepikhin,AravindhMahendran,DanielKeysers,MarioLucˇic´,Noam
Shazeer,AshishVaswani,andColinRaffelforusefuldiscussions.

## References

### AI Summary
The references section lists various papers that the authors cited in their work. These papers cover topics such as quantifying attention flow in transformers, learning representations by maximizing mutual information, and adaptive input representations for neural language modeling. The references provide background information and prior work that the authors built upon in their research.

### Original Content
SamiraAbnarandWillemZuidema. Quantifyingattentionflowintransformers. InACL,2020.
PhilipBachman,RDevonHjelm,andWilliamBuchwalter.Learningrepresentationsbymaximizing
mutualinformationacrossviews. InNeurIPS,2019.
wor
hctap
tupnI
ytiralimis
enisoC
)slexip(
ecnatsid
noitnetta
naeM
PublishedasaconferencepaperatICLR2021
AlexeiBaevskiandMichaelAuli. Adaptiveinputrepresentationsforneurallanguagemodeling. In

## Iclr,2019.

### AI Summary
This section lists several recent papers that explore the use of attention mechanisms and Transformers in computer vision tasks such as image recognition, object detection, and semantic segmentation. Key works include the use of attention to augment convolutional networks, the application of Transformers for end-to-end object detection, and unsupervised representation learning using contrastive methods. The papers also investigate the relationship between self-attention and convolutional layers, as well as the robustness and transferability of convolutional neural networks.

### Original Content
I.Bello,B.Zoph,Q.Le,A.Vaswani,andJ.Shlens. Attentionaugmentedconvolutionalnetworks.
InICCV,2019.
LucasBeyer,OlivierJ.He´naff,AlexanderKolesnikov,XiaohuaZhai,andAa¨ronvandenOord. Are
wedonewithimagenet? arXiv,2020.
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
ArvindNeelakantan,PranavShyam,GirishSastry,AmandaAskell,etal. Languagemodelsare
few-shotlearners. arXiv,2020.
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and
SergeyZagoruyko. End-to-endobjectdetectionwithtransformers. InECCV,2020.
Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. Generative pretraining from
pixels. InICML,2020a.
TingChen, SimonKornblith, MohammadNorouzi, andGeoffreyE.Hinton. Asimpleframework
forcontrastivelearningofvisualrepresentations. InICML,2020b.
Yen-ChunChen,LinjieLi,LichengYu,AhmedElKholy,FaisalAhmed,ZheGan,YuCheng,and
JingjingLiu. UNITER:UNiversalImage-TExtRepresentationLearning. InECCV,2020c.
RewonChild,ScottGray,AlecRadford,andIlyaSutskever. Generatinglongsequenceswithsparse
transformers. arXiv,2019.
Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self-
attentionandconvolutionallayers. InICLR,2020.
J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
imagedatabase. InCVPR,2009.
JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova. BERT:Pre-trainingofdeep
bidirectionaltransformersforlanguageunderstanding. InNAACL,2019.
Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander
Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan Moldovan, Sylvan
Gelly,NeilHoulsby,XiaohuaZhai,andMarioLucic. Onrobustnessandtransferabilityofconvo-
lutionalneuralnetworks. arXiv,2020.
KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun. Deepresiduallearningforimagerecog-
nition. InCVPR,2016.
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for
unsupervisedvisualrepresentationlearning. InCVPR,2020.
JonathanHo,NalKalchbrenner,DirkWeissenborn,andTimSalimans. Axialattentioninmultidi-
mensionaltransformers. arXiv,2019.
Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object
detection. InCVPR,2018.
HanHu,ZhengZhang,ZhendaXie,andStephenLin.Localrelationnetworksforimagerecognition.
InICCV,2019.
Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and
ThomasS.Huang. Ccnet: Criss-crossattentionforsemanticsegmentation. InICCV,2020.
OlivierJ.He´naff, AravindSrinivas, JeffreyDeFauw, AliRazavi, CarlDoersch, S.M.AliEslami,
andAaronvandenOord. Data-efficientimagerecognitionwithcontrastivepredictivecoding. In

## Icml,2020.

### AI Summary
This section appears to primarily be a list of references cited in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The references cover a range of topics related to deep learning for computer vision, including batch normalization, optimization methods like Adam, transfer learning, convolutional neural networks, and vision-language models. Without additional context about how these references relate to the main content of the paper, it is difficult to provide a meaningful summary of the key points.

### Original Content
PublishedasaconferencepaperatICLR2021
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducinginternalcovariateshift. 2015.
DiederikP.KingmaandJimmyBa. Adam: Amethodforstochasticoptimization. InICLR,2015.
Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly,
andNeilHoulsby. Bigtransfer(BiT):Generalvisualrepresentationlearning. InECCV,2020.
AlexKrizhevsky. Learningmultiplelayersoffeaturesfromtinyimages. Technicalreport,2009.
AlexKrizhevsky,IlyaSutskever,andGeoffreyE.Hinton. Imagenetclassificationwithdeepconvo-
lutionalneuralnetworks. InNIPS,2012.
Y.LeCun,B.Boser,J.Denker,D.Henderson,R.Howard,W.Hubbard,andL.Jackel. Backpropa-
gationappliedtohandwrittenzipcoderecognition. NeuralComputation,1:541–551,1989.
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,
MaximKrikun,NoamShazeer,andZhifengChen.Gshard:Scalinggiantmodelswithconditional
computationandautomaticsharding. arXiv,2020.
Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A
SimpleandPerformantBaselineforVisionandLanguage. InArxiv,2019.
FrancescoLocatello,DirkWeissenborn,ThomasUnterthiner,AravindhMahendran,GeorgHeigold,
JakobUszkoreit,AlexeyDosovitskiy,andThomasKipf. Object-centriclearningwithslotatten-
tion. arXiv,2020.
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visi-
olinguisticRepresentationsforVision-and-LanguageTasks. InNeurIPS.2019.
Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li,
Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised
pretraining. InECCV,2018.
M.NilsbackandA.Zisserman. Automatedflowerclassificationoveralargenumberofclasses. In

## Icvgip,2008.

### AI Summary
This section primarily consists of a list of references cited in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The references cover a range of topics related to computer vision, deep learning, self-supervised learning, and transformer models. Key works cited include papers on image recognition, video representation learning, panoptic segmentation, and the application of transformer architectures to various domains.

### Original Content
OmkarM.Parkhi,AndreaVedaldi,AndrewZisserman,andC.V.Jawahar. Catsanddogs. InCVPR,
2012.
NikiParmar,AshishVaswani,JakobUszkoreit,LukaszKaiser,NoamShazeer,AlexanderKu,and
DustinTran. Imagetransformer. InICML,2018.
B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM
Journal on Control and Optimization, 30(4):838–855, 1992. doi: 10.1137/0330046. URL
https://doi.org/10.1137/0330046.
SiyuanQiao,HuiyuWang,ChenxiLiu,WeiShen,andAlanYuille. Weightstandardization. arXiv
preprintarXiv:1903.10520,2019.
AlecRadford,KarthikNarasimhan,TimSalimans,andIlyaSutskever. Improvinglanguageunder-
standingwithunsupervisedlearning. TechnicalReport,2018.
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
modelsareunsupervisedmultitasklearners. TechnicalReport,2019.
PrajitRamachandran,NikiParmar,AshishVaswani,IrwanBello,AnselmLevskaya,andJonShlens.
Stand-aloneself-attentioninvisionmodels. InNeurIPS,2019.
Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable ef-
fectivenessofdataindeeplearningera. InICCV,2017.
ChenSun,AustinMyers,CarlVondrick,KevinMurphy,andCordeliaSchmid. Videobert: Ajoint
modelforvideoandlanguagerepresentationlearning. InICCV,2019.
PublishedasaconferencepaperatICLR2021
HugoTouvron,AndreaVedaldi,MatthijsDouze,andHerveJegou. Fixingthetrain-testresolution
discrepancy. InNeurIPS.2019.
HugoTouvron,AndreaVedaldi,MatthijsDouze,andHerveJegou. Fixingthetrain-testresolution
discrepancy: Fixefficientnet. arXivpreprintarXiv:2003.08237,2020.
Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain
Gelly, and Mario Lucic. Self-supervised learning of video-induced visual invariances. In Pro-
ceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition(CVPR),June
2020.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
ŁukaszKaiser,andIlliaPolosukhin. Attentionisallyouneed. InNIPS,2017.
Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
Axial-deeplab: Stand-aloneaxial-attentionforpanopticsegmentation. InECCV,2020a.
Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh
Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. arXiv preprint
arXiv:2003.07853,2020b.
Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao.
Learningdeeptransformermodelsformachinetranslation. InACL,2019.
Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In

## Cvpr,2018.

### AI Summary
DirkWeissenborn,OscarTa¨ckstro¨m,andJakobUszkoreit. Scalingautoregressivevideomodels. In

### Original Content
DirkWeissenborn,OscarTa¨ckstro¨m,andJakobUszkoreit. Scalingautoregressivevideomodels. In

## Iclr,2019.

### AI Summary
This section lists several recent papers that explore various techniques for improving image recognition performance, particularly using transformer-based architectures. Key approaches include using token-based image representations, self-supervised and semi-supervised learning methods, and studying self-attention mechanisms. The papers cited provide a snapshot of some of the latest advancements in pushing the state-of-the-art in image recognition.

### Original Content
BichenWu, ChenfengXu, XiaoliangDai, AlvinWan, PeizhaoZhang, MasayoshiTomizuka, Kurt
Keutzer,andPeterVajda. Visualtransformers: Token-basedimagerepresentationandprocessing
forcomputervision. arxiv,2020.
YuxinWuandKaimingHe. Groupnormalization. InECCV,2018.
Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student
improvesimagenetclassification. InCVPR,2020.
XiaohuaZhai,AvitalOliver,AlexanderKolesnikov,andLucasBeyer. S4L:Self-SupervisedSemi-
SupervisedLearning. InICCV,2019a.
Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario
Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A
large-scale study of representation learning with the visual task adaptation benchmark. arXiv
preprintarXiv:1910.04867,2019b.
HengshuangZhao,JiayaJia,andVladlenKoltun. Exploringself-attentionforimagerecognition. In

## Cvpr,2020.

### AI Summary
The table presents hyperparameters used for training various models, including Vision Transformers (ViT) and ResNets, on datasets such as JFT-300M, ImageNet-21k, and ImageNet. Key hyperparameters include the number of training epochs, base learning rate, learning rate decay schedule, weight decay, and dropout. All models were trained with a batch size of 4096 and a learning rate warmup of 10,000 steps, with an additional gradient clipping applied for ImageNet training.

### Original Content
PublishedasaconferencepaperatICLR2021
Models Dataset Epochs BaseLR LRdecay Weightdecay Dropout
ViT-B/{16,32} JFT-300M 7 8·10−4 linear 0.1 0.0
ViT-L/32 JFT-300M 7 6·10−4 linear 0.1 0.0
ViT-L/16 JFT-300M 7/14 4·10−4 linear 0.1 0.0
ViT-H/14 JFT-300M 14 3·10−4 linear 0.1 0.0
R50x{1,2} JFT-300M 7 10−3 linear 0.1 0.0
R101x1 JFT-300M 7 8·10−4 linear 0.1 0.0
R152x{1,2} JFT-300M 7 6·10−4 linear 0.1 0.0
R50+ViT-B/{16,32} JFT-300M 7 8·10−4 linear 0.1 0.0
R50+ViT-L/32 JFT-300M 7 2·10−4 linear 0.1 0.0
R50+ViT-L/16 JFT-300M 7/14 4·10−4 linear 0.1 0.0
ViT-B/{16,32} ImageNet-21k 90 10−3 linear 0.03 0.1
ViT-L/{16,32} ImageNet-21k 30/90 10−3 linear 0.03 0.1
ViT-∗ ImageNet 300 3·10−3 cosine 0.3 0.1
Table3: Hyperparametersfortraining. Allmodelsaretrainedwithabatchsizeof4096andlearn-
ing rate warmup of 10k steps. For ImageNet we found it beneficial to additionally apply gradient
clippingatglobalnorm1. Trainingresolutionis224.

## Appendix



## A Multihead Self-Attention

### AI Summary
Multihead self-attention (MSA) is an extension of the standard self-attention mechanism, which is a key component in neural architectures like Transformers. In MSA, multiple self-attention operations, called "heads," are run in parallel, and their outputs are concatenated and projected. To maintain constant computation and parameter count when varying the number of heads, the dimensionality of the query, key, and value representations is typically set to the total dimensionality divided by the number of heads.

### Original Content
Standardqkvself-attention(SA,Vaswanietal.(2017))isapopularbuildingblockforneuralarchi-
tectures. Foreachelementinaninputsequencez ∈ RN×D, wecomputeaweightedsumoverall
values v in the sequence. The attention weights A are based on the pairwise similarity between
twoelementsofthesequenceandtheirrespectivequeryqiandkeykj representations.
[q,k,v]=zU
qkv
qkv
∈RD×3Dh, (5)
(cid:16) (cid:112) (cid:17)
A=softmax qk(cid:62)/ D A∈RN×N, (6)
SA(z)=Av. (7)
Multiheadself-attention(MSA)isanextensionofSAinwhichwerunk self-attentionoperations,
called“heads”,inparallel,andprojecttheirconcatenatedoutputs. Tokeepcomputeandnumberof
parametersconstantwhenchangingk,D (Eq.5)istypicallysettoD/k.
MSA(z)=[SA 1(z);SA 2(z);··· ;SA k(z)]U
msa
msa
∈Rk·Dh×D (8)

## B Experiment Details



## B.1 Training

### AI Summary
The training setups for different models are summarized in Table 3. Strong regularization, including dropout applied after dense layers and positional embeddings, was found to be crucial when training models from scratch on ImageNet. Hybrid models follow the same training setup as their Vision Transformer (ViT) counterparts, and all training is performed at a resolution of 224.

### Original Content
Table 3 summarizes our training setups for our different models. We found strong regularization
to be key when training models from scratch on ImageNet. Dropout, when used, is applied after
every dense layer except for the the qkv-projections and directly after adding positional- to patch
embeddings. HybridmodelsaretrainedwiththeexactsetupastheirViTcounterparts. Finally,all
trainingisdoneonresolution224.

## B.1.1 Fine-Tuning

### AI Summary
This section describes the fine-tuning process for Vision Transformer (ViT) models using stochastic gradient descent (SGD) with a momentum of 0.9. The authors conduct a grid search over learning rates using small sub-splits of the training data as development sets, and then train the final models on the entire training set. The same setup is used for fine-tuning ResNets and hybrid models, with an additional learning rate value for ImageNet.

### Original Content
Wefine-tuneallViTmodelsusingSGDwithamomentumof0.9. Werunasmallgridsearchover
learningrates,seelearningraterangesinTable4.Todoso,weusesmallsub-splitsfromthetraining
set(10%forPetsandFlowers,2%forCIFAR,1%ImageNet)asdevelopmentsetandtrainonthe
remaining data. For final results we train on the entire training set and evaluate on the respective
test data. For fine-tuning ResNets and hybrid models we use the exact same setup, with the only
exceptionofImageNetwhereweaddanothervalue0.06tothelearningratesweep. Additionally,
PublishedasaconferencepaperatICLR2021
Dataset Steps BaseLR
ImageNet 20000 {0.003,0.01,0.03,0.06}

## Cifar100 10000 {0.001,0.003,0.01,0.03}



## Cifar10 10000 {0.001,0.003,0.01,0.03}

### AI Summary
This section discusses the hyperparameters used for fine-tuning Vision Transformer (ViT) models on various datasets, including Cifar10, Oxford-IIITPets, OxfordFlowers-102, and VTAB (19 tasks). When transferring ViT models to a new dataset, the entire head is replaced with a single, zero-initialized linear layer outputting the required number of classes for the target dataset. For VTAB, a learning rate of 0.01 and 2500 training steps were used, and it was found that ViT models benefit most from a high resolution (384×384) for all tasks.

### Original Content
Oxford-IIITPets 500 {0.001,0.003,0.01,0.03}
OxfordFlowers-102 500 {0.001,0.003,0.01,0.03}
VTAB(19tasks) 2500 0.01
Table4:Hyperparametersforfine-tuning.Allmodelsarefine-tunedwithcosinelearningratedecay,
abatchsizeof512,noweightdecay,andgradclippingatglobalnorm1.Ifnotmentionedotherwise,
fine-tuningresolutionis384.
for ResNets we also run the setup of Kolesnikov et al. (2020) and select the best results across
thisrunandoursweep. Finally,ifnotmentionedotherwise,allfine-tuningexperimentsrunat384
resolution(runningfine-tuningatdifferentresolutionthantrainingiscommonpractice(Kolesnikov
etal.,2020)).
WhentransferringViTmodelstoanotherdataset,weremovethewholehead(twolinearlayers)and
replaceitbyasingle,zero-initializedlinearlayeroutputtingthenumberofclassesrequiredbythe
targetdataset. Wefoundthistobealittlemorerobustthansimplyre-initializingtheverylastlayer.
For VTAB we follow the protocol in Kolesnikov et al. (2020), and use the same hyperparameter
settingforalltasks. Weusealearningrateof0.01andtrainfor2500steps(Tab.4). Wechosethis
settingbyrunningasmallsweepovertwolearningratesandtwoschedules,andselectingthesetting
withthehighestVTABscoreonthe200-examplevalidationsets.Wefollowthepre-processingused
inKolesnikovetal.(2020),exceptthatwedonotusetask-specificinputresolutions.Insteadwefind
thatVisionTransformerbenefitsmostfromahighresolution(384×384)foralltasks.

## B.1.2 Self-Supervision

### AI Summary
This section describes experiments with self-supervised pretraining for image recognition using a masked patch prediction objective. The authors corrupt 50% of patch embeddings and predict the mean color of the corrupted patches. They find that predicting only the mean 3-bit color works best, and note that large datasets and long pretraining are not necessarily required to achieve performance gains on downstream tasks like ImageNet classification.

### Original Content
Weemploythemaskedpatchpredictionobjectiveforpreliminaryself-supervisionexperiments. To
do so we corrupt 50% of patch embeddings by either replacing their embeddings with a learnable
[mask] embedding (80%), a random other patch embedding (10%) or just keeping them as is
(10%). ThissetupisverysimilartotheoneusedforlanguagebyDevlinetal.(2019). Finally,we
predictthe3-bit,meancolor(i.e.,512colorsintotal)ofeverycorruptedpatchusingtheirrespective
patchrepresentations.
Wetrainedourself-supervisedmodelfor1Msteps(ca. 14epochs)withbatchsize4096onJFT.We
useAdam,withabaselearningrateof2·10−4,warmupof10kstepsandcosinelearningratedecay.
As prediction targets for pretraining we tried the following settings: 1) predicting only the mean,
3bitcolor(i.e.,1predictionof512colors),2)predictinga4×4downsizedversionofthe16×16
patchwith3bitcolorsinparallel(i.e.,16predictionsof512colors),3)regressiononthefullpatch
usingL2(i.e.,256regressionsonthe3RGBchannels). Surprisingly,wefoundthatallworkedquite
well,thoughL2wasslightlyworse. Wereportfinalresultsonlyforoption1)becauseithasshown
bestfew-shotperformance. Wealsoexperimentedwith15%corruptionrateasusedbyDevlinetal.
(2019)butresultswerealsoslightlyworseonourfew-shotmetrics.
Lastly, we would like to remark that our instantiation of masked patch prediction doesn’t require
such an enormous amount of pretraining nor a large dataset such as JFT in order to lead to sim-
ilar performance gains on ImageNet classification. That is, we observed diminishing returns on
downstream performance after 100k pretraining steps, and see similar gains when pretraining on
ImageNet.

## C Additional Results

### AI Summary
The experiments demonstrate that Vision Transformers (ViT) pre-trained on larger datasets (ImageNet-21k and JFT-300M) achieve superior transfer learning performance compared to those pre-trained on ImageNet alone, with top-1 accuracy improvements of up to 5% on various datasets. Scaling up ViT models in terms of depth and width, as well as using hybrid models combining ResNet and ViT, leads to further performance gains, surpassing ResNet baselines by a significant margin while requiring comparable pre-training compute. The results highlight the potential of transformer-based architectures for image recognition tasks when trained on large-scale datasets.

### Original Content
Wereportdetailedresultscorrespondingtothefigurespresentedinthepaper. Table5corresponds
to Figure 3 from the paper and shows transfer performance of different ViT models pre-trained
on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. Table 6 corresponds to
PublishedasaconferencepaperatICLR2021
ViT-B/16 ViT-B/32 ViT-L/16 ViT-L/32 ViT-H/14
ImageNet CIFAR-10 98.13 97.77 97.86 97.94 -
CIFAR-100 87.13 86.31 86.35 87.07 -
ImageNet 77.91 73.38 76.53 71.16 -
ImageNetReaL 83.57 79.56 82.19 77.83 -
OxfordFlowers-102 89.49 85.43 89.66 86.36 -
Oxford-IIIT-Pets 93.81 92.04 93.64 91.35 -
ImageNet-21k CIFAR-10 98.95 98.79 99.16 99.13 99.27
CIFAR-100 91.67 91.97 93.44 93.04 93.82
ImageNet 83.97 81.28 85.15 80.99 85.13
ImageNetReaL 88.35 86.63 88.40 85.65 88.70
OxfordFlowers-102 99.38 99.11 99.61 99.19 99.51
Oxford-IIIT-Pets 94.43 93.02 94.73 93.09 94.82
JFT-300M CIFAR-10 99.00 98.61 99.38 99.19 99.50
CIFAR-100 91.87 90.49 94.04 92.52 94.55
ImageNet 84.15 80.73 87.12 84.37 88.04
ImageNetReaL 88.85 86.27 89.99 88.28 90.33
OxfordFlowers-102 99.56 99.27 99.56 99.45 99.68
Oxford-IIIT-Pets 95.80 93.40 97.11 95.83 97.56
Table5: Top1accuracy(in%)ofVisionTransformeronvariousdatasetswhenpre-trainedonIm-
ageNet,ImageNet-21korJFT300M.ThesevaluescorrespondtoFigure3inthemaintext. Models
are fine-tuned at 384 resolution. Note that the ImageNet results are computed without additional
techniques(Polyakaveragingand512resolutionimages)usedtoachieveresultsinTable2.
Epochs ImageNet ImageNetReaL CIFAR-10 CIFAR-100 Pets Flowers exaFLOPs
name
ViT-B/32 7 80.73 86.27 98.61 90.49 93.40 99.27 55
ViT-B/16 7 84.15 88.85 99.00 91.87 95.80 99.56 224
ViT-L/32 7 84.37 88.28 99.19 92.52 95.83 99.45 196
ViT-L/16 7 86.30 89.43 99.38 93.46 96.81 99.66 783
ViT-L/16 14 87.12 89.99 99.38 94.04 97.11 99.56 1567
ViT-H/14 14 88.08 90.36 99.50 94.71 97.11 99.71 4262
ResNet50x1 7 77.54 84.56 97.67 86.07 91.11 94.26 50
ResNet50x2 7 82.12 87.94 98.29 89.20 93.43 97.02 199
ResNet101x1 7 80.67 87.07 98.48 89.17 94.08 95.95 96
ResNet152x1 7 81.88 87.96 98.82 90.22 94.17 96.94 141
ResNet152x2 7 84.97 89.69 99.06 92.05 95.37 98.62 563
ResNet152x2 14 85.56 89.89 99.24 91.92 95.75 98.75 1126
ResNet200x3 14 87.22 90.15 99.34 93.53 96.32 99.04 3306
R50x1+ViT-B/32 7 84.90 89.15 99.01 92.24 95.75 99.46 106
R50x1+ViT-B/16 7 85.58 89.65 99.14 92.63 96.65 99.40 274
R50x1+ViT-L/32 7 85.68 89.04 99.24 92.93 96.97 99.43 246
R50x1+ViT-L/16 7 86.60 89.72 99.18 93.64 97.03 99.40 859
R50x1+ViT-L/16 14 87.12 89.76 99.31 93.89 97.36 99.11 1668
Table6: Detailedresultsofmodelscalingexperiments. ThesecorrespondtoFigure5inthemain
paper. We show transfer accuracy on several datasets, as well as the pre-training compute (in ex-
aFLOPs).
Figure5fromthepaperandshowsthetransferperformanceofViT,ResNet,andhybridmodelsof
varyingsize,aswellastheestimatedcomputationalcostoftheirpre-training.

## D Additional Analyses



## D.1 Sgdvs. Adamforresnets

### AI Summary
This section compares the performance of ResNet models pre-trained using Adam and SGD optimizers. Experiments show that pre-training with Adam outperforms SGD on most datasets and on average, justifying the unconventional choice of Adam as the optimizer for pre-training ResNets on the JFT dataset. However, the absolute performance numbers are lower than those reported in previous work due to a shorter pre-training period of 7 epochs instead of 30.

### Original Content
ResNetsaretypicallytrainedwithSGDandouruseofAdamasoptimizerisquiteunconventional.
Here we show the experiments that motivated this choice. Namely, we compare the fine-tuning
PublishedasaconferencepaperatICLR2021
ResNet50 ResNet152x2
Dataset Adam SGD Adam SGD
ImageNet 77.54 78.24 84.97 84.37
CIFAR10 97.67 97.46 99.06 99.07
CIFAR100 86.07 85.17 92.05 91.06
Oxford-IIITPets 91.11 91.00 95.37 94.79
OxfordFlowers-102 94.26 92.06 98.62 99.32
Average 89.33 88.79 94.01 93.72
Table7: Fine-tuningResNetmodelspre-trainedwithAdamandSGD.
0.8
0.6
0.7 0.5
0.4 Models 0.6 Models
All All
0.3 Depth Depth
Patch size 0.5 Patch size
0.2 Width MLP Width MLP
Width Width
0.4
100 101 100 101
Relative Compute Relative Compute
Figure8: ScalingdifferentmodeldimensionsoftheVisionTransformer.
performance of two ResNets – 50x1 and 152x2 – pre-trained on JFT with SGD and Adam. For
SGD,weusethehyperparametersrecommendedbyKolesnikovetal.(2020). Resultsarepresented
in Table 7. Adam pre-training outperforms SGD pre-training on most datasets and on average.
ThisjustifiesthechoiceofAdamastheoptimizerusedtopre-trainResNetsonJFT.Notethatthe
absolutenumbersarelowerthanthosereportedbyKolesnikovetal.(2020),sincewepre-trainonly
for7epochs,not30.

## D.2 Transformershape

### AI Summary
This section explores how scaling different dimensions of the Transformer architecture affects performance on the ImageNet dataset. The authors found that increasing the model's depth (number of layers) yields the most significant improvements, while increasing width results in the smallest gains. They also observed that decreasing the patch size, which effectively increases the sequence length, leads to robust improvements without adding parameters, suggesting that compute might be a better predictor of performance than the number of parameters.

### Original Content
WeranablationsonscalingdifferentdimensionsoftheTransformerarchitecturetofindoutwhich
are best suited for scaling to very large models. Figure 8 shows 5-shot performance on ImageNet
fordifferentconfigurations. AllconfigurationsarebasedonaViTmodelwith8layers,D = 1024,
D = 2048 and a patch size of 32, the intersection of all lines. We can see that scaling the
MLP
depth results in the biggest improvements which are clearly visible up until 64 layers. However,
diminishing returns are already visible after 16 layers. Interestingly, scaling the width of the net-
work seems to result in the smallest changes. Decreasing the patch size and thus increasing the
effectivesequencelengthshowssurprisinglyrobustimprovementswithoutintroducingparameters.
Thesefindingssuggestthatcomputemightbeabetterpredictorofperformancethanthenumberof
parameters,andthatscalingshouldemphasizedepthoverwidthifany. Overall,wefindthatscaling
alldimensionsproportionallyresultsinrobustimprovements.

## D.3 Headtypeand Class Token

### AI Summary
This section compares using a class token versus global average pooling (GAP) for the image classification output in the Vision Transformer model. While initial attempts using only GAP performed poorly, the authors found that both the class token approach and GAP worked similarly well, but required different learning rates to achieve optimal performance. The class token design, inherited from the original Transformer model for text, was used throughout the main paper.

### Original Content
InordertostayascloseaspossibletotheoriginalTransformermodel,wemadeuseofanadditional
[class] token, which is taken as image representation. The output of this token is then trans-
formedintoaclasspredictionviaasmallmulti-layerperceptron(MLP)withtanhasnon-linearity
inthesinglehiddenlayer.
This design is inherited from the Transformer model for text, and we use it throughout the main
paper. An initial attempt at using only image-patch embeddings, globally average-pooling (GAP)
them,followedbyalinearclassifier—justlikeResNet’sfinalfeaturemap—performedverypoorly.
However, we found that this is neither due to the extra token, nor to the GAP operation. Instead,
tohs5
teNegamI
tohs5
egarevA
PublishedasaconferencepaperatICLR2021
CLS-Token, lr=8e-4
55 GAP, lr=8e-4
GAP, lr=3e-4
0 1 2 3 4 5 6 7
Epochs of training
Figure 9: Comparison of class-token and global average pooling classifiers. Both work similarly
well,butrequiredifferentlearning-rates.
Pos. Emb. Default/Stem EveryLayer EveryLayer-Shared
NoPos. Emb. 0.61382 N/A N/A
1-DPos. Emb. 0.64206 0.63964 0.64292
2-DPos. Emb. 0.64001 0.64046 0.64022
Rel. Pos. Emb. 0.64032 N/A N/A
Table8: ResultsoftheablationstudyonpositionalembeddingswithViT-B/16modelevaluatedon
ImageNet5-shotlinear.
thedifferenceinperformanceisfullyexplainedbytherequirementforadifferentlearning-rate,see
Figure9.

## D.4 Positionalembedding

### AI Summary
This section explores different methods for encoding spatial information in the Vision Transformer (ViT) model using positional embeddings. The authors experimented with 1D and 2D positional embeddings, relative positional embeddings, and different ways of incorporating these embeddings into the model. Their results show that while including positional information is crucial for performance, the specific method of encoding spatial information has little impact, likely because the Transformer operates on patch-level rather than pixel-level inputs.

### Original Content
Weranablationsondifferentwaysofencodingspatialinformationusingpositionalembedding. We
triedthefollowingcases:
• Providingnopositionalinformation: Consideringtheinputsasabagofpatches.
• 1-dimensional positional embedding: Considering the inputs as a sequence of patches in
therasterorder(defaultacrossallotherexperimentsinthispaper).
• 2-dimensional positional embedding: Considering the inputs as a grid of patches in two
dimensions. In this case, two sets of embeddings are learned, each for one of the axes,
X-embedding,andY-embedding,eachwithsizeD/2. Then,basedonthecoordinateon
the path in the input, we concatenate the X and Y embedding to get the final positional
embeddingforthatpatch.
• Relativepositionalembeddings: Consideringtherelativedistancebetweenpatchestoen-
code the spatial information as instead of their absolute position. To do so, we use 1-
dimensionalRelativeAttention,inwhichwedefinetherelativedistanceallpossiblepairs
ofpatches. Thus, foreverygivenpair(oneasquery, andtheotheraskey/valueintheat-
tention mechanism), we have an offset p −p , where each offset is associated with an
q k
embedding. Then, we simply run extra attention, where we use the original query (the
content of query), but use relative positional embeddings as keys. We then use the log-
its from the relative attention as a bias term and add it to the logits of the main attention
(content-basedattention)beforeapplyingthesoftmax.
In addition to different ways of encoding spatial information, we also tried different ways of in-
corporating this information in our model. For the 1-dimensional and 2-dimensional positional
embeddings, wetriedthreedifferentcases: (1)addpositionalembeddingstotheinputsrightafter
ycarucca
tohs-5
raenil
teNegamI
PublishedasaconferencepaperatICLR2021
ViT-L16 ViT-L16 ViT-L16
7 epochs, LR=0.0002, WD=0.01 7 epochs, LR=0.0004, WD=0.1 14 epochs, LR=0.0004, WD=0.1
1 1 1
1 1 1
2 2 2
3 3 3
4 4 4
5 5 5
6 6 6
7 7 7
8 8 8
9 9 9
10 10 10
11 11 11
12 12 12
13 13 13
14 14 14
1 2 3 4 5 6 7 8 9 1011121314 1 1 2 3 4 5 6 7 8 9 1011121314 1 1 2 3 4 5 6 7 8 9 1011121314 1
Input patch column Input patch column Input patch column
Figure10: Positionembeddingsofmodelstrainedwithdifferenthyperparameters.
the stem of them model and before feeding the inputs to the Transformer encoder (default across
all other experiments in this paper); (2) learn and add positional embeddings to the inputs at the
beginningofeachlayer; (3)addalearnedpositionalembeddingstotheinputsatthebeginningof
eachlayer(sharedbetweenlayers).
Table8summarizestheresultsfromthisablationstudyonaViT-B/16model. Aswecansee,while
thereisalargegapbetweentheperformancesofthemodelwithnopositionalembeddingandmod-
els with positional embedding, there is little to no difference between different ways of encoding
positional information. We speculate that since our Transformer encoder operates on patch-level
inputs,asopposedtopixel-level,thedifferencesinhowtoencodespatialinformationislessimpor-
tant. Moreprecisely,inpatch-levelinputs,thespatialdimensionsaremuchsmallerthantheoriginal
pixel-levelinputs, e.g., 14×14asopposedto224×224, andlearningtorepresentthespatialre-
lationsinthisresolutionisequallyeasyforthesedifferentpositionalencodingstrategies. Evenso,
thespecificpatternofpositionembeddingsimilaritylearnedbythenetworkdependsonthetraining
hyperparameters(Figure10).
ViT-L/16 R50x1 + ViT-L/16
120 120
100 100
80 80
60 60
40 40
Head 1 Head 1
20 Head 2 20 Head 2
Head 3 Head 3
... ...
0 0
0 5 10 15 20 0 5 10 15 20
Network depth (layer) Network depth (layer)
Figure11: Sizeofattendedareabyheadandnetworkdepth. Attentiondistancewascomputedfor
128exampleimagesbyaveragingthedistancebetweenthequerypixelandallotherpixels,weighted
by the attention weight. Each dot shows the mean attention distance across images for one of 16
headsatonelayer. Imagewidthis224pixels.

## D.5 Empiricalcomputationalcosts

### AI Summary
This section empirically evaluates the computational costs of Vision Transformer (ViT) models compared to ResNet models on TPUv3 accelerators. The results show that ViT models have comparable inference speed to similar ResNets across various input sizes, with the theoretical bi-quadratic scaling only starting to happen for the largest ViT models at high resolutions. Furthermore, large ViT models are found to be more memory-efficient than ResNets, being able to fit larger batch sizes per core on the accelerator.

### Original Content
Wearealsointerestedinreal-worldspeedofthearchitecturesonourhardware,whichisnotalways
wellpredictedbytheoreticalFLOPsduetodetailslikelanewidthsandcachesizes.Forthispurpose,
wor
hctap
tupnI
)slexip(
ecnatsid
noitnetta
naeM
ywtiorra
lhimctiasp
e tnuisponCI
ywtiorra
lhimctiasp
e tnuisponCI
ytiralimis
enisoC
PublishedasaconferencepaperatICLR2021
weperformtimingofinferencespeedforthemainmodelsofinterest,onaTPUv3accelerator;the
differencebetweeninferenceandbackpropspeedisaconstantmodel-independentfactor.
Figure12(left)showshowmanyimagesonecorecanhandlepersecond,acrossvariousinputsizes.
Everysinglepointreferstothepeakperformancemeasuredacrossawiderangeofbatch-sizes. As
canbeseen,thetheoreticalbi-quadraticscalingofViTwithimagesizeonlybarelystartshappening
forthelargestmodelsatthelargestresolutions.
Another quantity of interest is the largest batch-size each model can fit onto a core, larger being
betterforscalingtolargedatasets. Figure12(right)showsthisquantityforthesamesetofmodels.
ThisshowsthatlargeViTmodelshaveaclearadvantageintermsofmemory-efficiencyoverResNet
models.
R50x1 ViT-B/32 ViT-B/16 ViT-H/14
R50x2 ViT-L/32 ViT-L/16 R152x4
64 128 224 384 512 64 128 224 384 512
Input size [px] Input size [px]
Figure 12: Left: Real wall-clock timings of various architectures across input sizes. ViT models
havespeedcomparabletosimilarResNets. Right:Largestper-corebatch-sizefittingondevicewith
variousarchitecturesacrossinputsizes. ViTmodelsareclearlymorememory-efficient.

## D.6 Axialattention

### AI Summary
Axial attention is a technique that performs self-attention operations along individual axes of a multidimensional input tensor, allowing for efficient processing of large inputs. The authors implemented Axial-ViT models, which replace the global self-attention in ViT Transformer blocks with row and column self-attention, and AxialResNet, which replaces 3x3 convolutions in ResNet50 with axial self-attention. While Axial-ViT models outperform their ViT counterparts on ImageNet 5-shot linear accuracy, they come with increased computational costs, and the naive implementation of AxialResNet is slow on TPUs despite its reasonable accuracy/compute trade-off.

### Original Content
AxialAttention(Huangetal.,2020;Hoetal.,2019)isasimple,yeteffectivetechniquetorunself-
attentiononlargeinputsthatareorganizedasmultidimensionaltensors. Thegeneralideaofaxial
attention is to perform multiple attention operations, each along a single axis of the input tensor,
insteadofapplying1-dimensionalattentiontotheflattenedversionoftheinput. Inaxialattention,
eachattentionmixesinformationalongaparticularaxis,whilekeepinginformationalongtheother
axesindependent. Alongthisline,Wangetal.(2020b)proposedtheAxialResNetmodelinwhich
all the convolutions with kernel size 3×3 in a ResNet50 are replaced by axial self-attention, i.e.
a row and column attention, augmented by relative positional encoding. We have implemented
AxialResNetasabaselinemodel.3.
Moreover, we have modified ViT to process inputs in the 2-dimensional shape, instead of a 1-
dimensional sequence of patches, and incorporate Axial Transformer blocks, in which instead of
a self-attention followed by an MLP, we have a a row-self-attention plus an MLP followed by a
column-self-attentionplusanMLP.
Figure13,presenttheperformanceofAxialResNet,Axial-ViT-B/32andAxial-ViT-B/16onIma-
geNet5shotlinear,whenpretrainedonJFTdataset,versesthepretrainingcompute,bothintermsof
numberofFLOPsandinferencetime(exampleperseconds). Aswecansee,bothAxial-ViT-B/32
andAxial-ViT-B/16dobetterthantheirViT-Bcounterpartintermsofperformance,butitcomesat
3Ourimplementationisbasedontheopen-sourcedPyTorchimplementationinhttps://github.com/
csrhddlam/axial-deeplab. In our experiments, we reproduced the scores reported in (Wang et al.,
2020b)intermsofaccuracy,however,ourimplementation,similartotheopen-sourceimplementation,isvery
slow on TPUs. Therefore, we were not able to use it for extensive large-scale experiments. These may be
unlockedbyacarefullyoptimizedimplementation.
]eroc/ces/gmi[
deeps
ecnerefni
kaeP
ezis-hctab
eroc-rep
tsegraL
PublishedasaconferencepaperatICLR2021
AxialViT-B/16 AxialViT-B/16
0.650 0.650
ViT-B/16 ViT-B/16
0.625 0.625
0.600 0.600
AxialViT-B/32 AxialViT-B/32
0.575 0.575
ViT-B/32 ViT-B/32
AxialResNet50 AxialResNet50
0.550 0.550
0.525 0.525
0.500 ResNet50 0.500 ResNet50
102 103 102
Total compute [exaFLOPs] Peak inference speed [img/sec/core]
Figure13: PerformanceofAxial-Attentionbasedmodels,intermsoftop-1accuracyonImageNet
5-shotlinear,versustheirspeedintermsofnumberofFLOPs(left)andinferencetime(left).
thecostofmorecompute.ThisisbecauseinAxial-ViTmodels,eachTransformerblockwithglobal
self-attentionisreplacedbytwoAxialTransformerblocks,onewithrowandonewithcolumnself-
attention and although the sequence length that self-attention operates on is smaller in axial case,
there is a extra MLP per Axial-ViT block. For the AxialResNet, although it looks reasonable in
terms of accuracy/compute trade-off (Figure 13, left), the naive implementation is extremely slow
onTPUs(Figure13,right).

## D.7 Attentiondistance

### AI Summary
This section analyzes how the self-attention mechanism in the Vision Transformer (ViT) model integrates information across an image at different layers. The average distance spanned by attention weights, termed "attention distance," varies significantly across attention heads in lower layers, with some attending to most of the image while others focus on smaller regions. As the network depth increases, the attention distance increases for all heads, and in the second half of the network, most heads attend widely across tokens, enabling the model to capture global context.

### Original Content
To understand how ViT uses self-attention to integrate information across the image, we analyzed
the average distance spanned by attention weights at different layers (Figure 11). This “attention
distance”isanalogoustoreceptivefieldsizeinCNNs. Averageattentiondistanceishighlyvariable
acrossheadsinlowerlayers,withsomeheadsattendingtomuchoftheimage,whileothersattend
tosmallregionsatornearthequerylocation. Asdepthincreases,attentiondistanceincreasesforall
heads. Inthesecondhalfofthenetwork,mostheadsattendwidelyacrosstokens.

## D.8 Attentionmaps

### AI Summary
In this section, the authors describe their method for generating attention maps that visualize how the output token attends to different regions of the input image. They employ the Attention Rollout technique, which involves averaging attention weights across all heads in the ViT-L/16 model and recursively multiplying the weight matrices from all layers. This process accounts for the mixing of attention across tokens throughout the entire network, providing insights into the model's focus when making predictions.

### Original Content
Tocomputemapsoftheattentionfromtheoutputtokentotheinputspace(Figures6and14), we
used Attention Rollout (Abnar & Zuidema, 2020). Briefly, we averaged attention weights of ViT-
L/16acrossallheadsandthenrecursivelymultipliedtheweightmatricesofalllayers.Thisaccounts
forthemixingofattentionacrosstokensthroughalllayers.

## D.9 Objectnetresults

### AI Summary
WealsoevaluateourflagshipViT-H/14modelontheObjectNetbenchmarkfollowingtheevaluation
setupinKolesnikovetal.(2020),resultingin82.1%top-5accuracyand61.7%top-1accuracy.

### Original Content
WealsoevaluateourflagshipViT-H/14modelontheObjectNetbenchmarkfollowingtheevaluation
setupinKolesnikovetal.(2020),resultingin82.1%top-5accuracyand61.7%top-1accuracy.

## D.10 Vtabbreakdown

### AI Summary
Table 9 shows the performance breakdown of different Vision Transformer (ViT) models across various tasks in the Visual Task Adaptation Benchmark (VTAB-1k). The ViT-H/14 model pre-trained on the JFT dataset achieves the highest mean accuracy of 77.6% across the 20 tasks, outperforming the ViT-L/16 models pre-trained on JFT and ImageNet-21k (I21k) which attain mean accuracies of 76.3% and 72.7% respectively.

### Original Content
Table9showsthescoresattainedoneachoftheVTAB-1ktasks.
ycarucca
1-pot
raenil
tohs-5
teNegamI
ycarucca
1-pot
raenil
tohs-5
teNegamI
PublishedasaconferencepaperatICLR2021
1 2 3 4 5 6 7 8
9 10 11 12 13 14 15 16
17 18 19 20 21 22 23 24
25 26 27 28 29 30 31 32
33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48
49 50 51 52 53 54 55 56
57 58 59 60 61 62 63 64
65 66 67 68 69 70 71 72
73 74 75 76 77 78 79 80
81 82 83 84 85 86 87 88
89 90 91 92 93 94 95 96
97 98 99 100 101 102 103 104
105 106 107 108 109 110 111 112
113 114 115 116 117 118 119 120
121 122 123 124 125 126 127 128
Figure14: FurtherexampleattentionmapsasinFigure6(randomselection).
PublishedasaconferencepaperatICLR2021
Table9: BreakdownofVTAB-1kperformanceacrosstasks.
ViT-H/14(JFT) 95.3 85.5 75.2 99.7 97.2 65.0 88.9 83.3 96.7 91.4 76.6 91.7 63.8 53.1 79.4 63.3 84.5 33.2 51.2 77.6
ViT-L/16(JFT) 95.4 81.9 74.3 99.7 96.7 63.5 87.4 83.6 96.5 89.7 77.1 86.4 63.1 49.7 74.5 60.5 82.2 36.2 51.1 76.3
ViT-L/16(I21k)90.8 84.1 74.1 99.3 92.7 61.0 80.9 82.5 95.6 85.2 75.3 70.3 56.1 41.9 74.7 64.9 79.9 30.5 41.7 72.7
101hcetlaC 001-RAFIC
DTD
201srewolF
steP
793nuS NHVS noylemaC TASoruE 54csiseR yhtaponiteR tnuoC-rvelC tsiD-rvelC baLMD coL-rpSd irO-rpSd tsiD-ITTIK mizA-BRONs velE-BRONs
naeM

## References

1. SamiraAbnarandWillemZuidema. Quantifyingattentionflowintransformers. InACL,2020. PhilipBachman,RDevonHjelm,andWilliamBuchwalter.Learningrepresentationsbymaximizing mutualinformationacrossviews. InNeurIPS,2019. 9 wor hctap tupnI ytiralimis enisoC )slexip( ecnatsid noitnetta naeM PublishedasaconferencepaperatICLR2021 AlexeiBaevskiandMichaelAuli. Adaptiveinputrepresentationsforneurallanguagemodeling. In ICLR,2019. I.Bello,B.Zoph,Q.Le,A.Vaswani,andJ.Shlens. Attentionaugmentedconvolutionalnetworks. InICCV,2019. LucasBeyer,OlivierJ.He´naff,AlexanderKolesnikov,XiaohuaZhai,andAa¨ronvandenOord. Are wedonewithimagenet? arXiv,2020. Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, ArvindNeelakantan,PranavShyam,GirishSastry,AmandaAskell,etal. Languagemodelsare few-shotlearners. arXiv,2020. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and SergeyZagoruyko. End-to-endobjectdetectionwithtransformers. InECCV,2020. Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. Generative pretraining from pixels. InICML,2020a. TingChen, SimonKornblith, MohammadNorouzi, andGeoffreyE.Hinton. Asimpleframework forcontrastivelearningofvisualrepresentations. InICML,2020b. Yen-ChunChen,LinjieLi,LichengYu,AhmedElKholy,FaisalAhmed,ZheGan,YuCheng,and JingjingLiu. UNITER:UNiversalImage-TExtRepresentationLearning. InECCV,2020c. RewonChild,ScottGray,AlecRadford,andIlyaSutskever. Generatinglongsequenceswithsparse transformers. arXiv,2019. Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self- attentionandconvolutionallayers. InICLR,2020. J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical imagedatabase. InCVPR,2009. JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova. BERT:Pre-trainingofdeep bidirectionaltransformersforlanguageunderstanding. InNAACL,2019. Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan Moldovan, Sylvan Gelly,NeilHoulsby,XiaohuaZhai,andMarioLucic. Onrobustnessandtransferabilityofconvo- lutionalneuralnetworks. arXiv,2020. KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun. Deepresiduallearningforimagerecog- nition. InCVPR,2016. Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervisedvisualrepresentationlearning. InCVPR,2020. JonathanHo,NalKalchbrenner,DirkWeissenborn,andTimSalimans. Axialattentioninmultidi- mensionaltransformers. arXiv,2019. Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. InCVPR,2018. HanHu,ZhengZhang,ZhendaXie,andStephenLin.Localrelationnetworksforimagerecognition. InICCV,2019. Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and ThomasS.Huang. Ccnet: Criss-crossattentionforsemanticsegmentation. InICCV,2020. OlivierJ.He´naff, AravindSrinivas, JeffreyDeFauw, AliRazavi, CarlDoersch, S.M.AliEslami, andAaronvandenOord. Data-efficientimagerecognitionwithcontrastivepredictivecoding. In ICML,2020. 10 PublishedasaconferencepaperatICLR2021 Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducinginternalcovariateshift. 2015. DiederikP.KingmaandJimmyBa. Adam: Amethodforstochasticoptimization. InICLR,2015. Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, andNeilHoulsby. Bigtransfer(BiT):Generalvisualrepresentationlearning. InECCV,2020. AlexKrizhevsky. Learningmultiplelayersoffeaturesfromtinyimages. Technicalreport,2009. AlexKrizhevsky,IlyaSutskever,andGeoffreyE.Hinton. Imagenetclassificationwithdeepconvo- lutionalneuralnetworks. InNIPS,2012. Y.LeCun,B.Boser,J.Denker,D.Henderson,R.Howard,W.Hubbard,andL.Jackel. Backpropa- gationappliedtohandwrittenzipcoderecognition. NeuralComputation,1:541–551,1989. Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, MaximKrikun,NoamShazeer,andZhifengChen.Gshard:Scalinggiantmodelswithconditional computationandautomaticsharding. arXiv,2020. Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A SimpleandPerformantBaselineforVisionandLanguage. InArxiv,2019. FrancescoLocatello,DirkWeissenborn,ThomasUnterthiner,AravindhMahendran,GeorgHeigold, JakobUszkoreit,AlexeyDosovitskiy,andThomasKipf. Object-centriclearningwithslotatten- tion. arXiv,2020. Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visi- olinguisticRepresentationsforVision-and-LanguageTasks. InNeurIPS.2019. Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. InECCV,2018. M.NilsbackandA.Zisserman. Automatedflowerclassificationoveralargenumberofclasses. In ICVGIP,2008. OmkarM.Parkhi,AndreaVedaldi,AndrewZisserman,andC.V.Jawahar. Catsanddogs. InCVPR,
2. 2012. NikiParmar,AshishVaswani,JakobUszkoreit,LukaszKaiser,NoamShazeer,AlexanderKu,and DustinTran. Imagetransformer. InICML,2018. B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838–855, 1992. doi: 10.1137/0330046. URL https://doi.org/10.1137/0330046. SiyuanQiao,HuiyuWang,ChenxiLiu,WeiShen,andAlanYuille. Weightstandardization. arXiv preprintarXiv:1903.10520,2019. AlecRadford,KarthikNarasimhan,TimSalimans,andIlyaSutskever. Improvinglanguageunder- standingwithunsupervisedlearning. TechnicalReport,2018. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language modelsareunsupervisedmultitasklearners. TechnicalReport,2019. PrajitRamachandran,NikiParmar,AshishVaswani,IrwanBello,AnselmLevskaya,andJonShlens. Stand-aloneself-attentioninvisionmodels. InNeurIPS,2019. Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable ef- fectivenessofdataindeeplearningera. InICCV,2017. ChenSun,AustinMyers,CarlVondrick,KevinMurphy,andCordeliaSchmid. Videobert: Ajoint modelforvideoandlanguagerepresentationlearning. InICCV,2019. 11 PublishedasaconferencepaperatICLR2021 HugoTouvron,AndreaVedaldi,MatthijsDouze,andHerveJegou. Fixingthetrain-testresolution discrepancy. InNeurIPS.2019. HugoTouvron,AndreaVedaldi,MatthijsDouze,andHerveJegou. Fixingthetrain-testresolution discrepancy: Fixefficientnet. arXivpreprintarXiv:2003.08237,2020. Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. Self-supervised learning of video-induced visual invariances. In Pro- ceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition(CVPR),June
3. 2020. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, ŁukaszKaiser,andIlliaPolosukhin. Attentionisallyouneed. InNIPS,2017. Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-aloneaxial-attentionforpanopticsegmentation. InECCV,2020a. Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. arXiv preprint arXiv:2003.07853,2020b. Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learningdeeptransformermodelsformachinetranslation. InACL,2019. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR,2018. DirkWeissenborn,OscarTa¨ckstro¨m,andJakobUszkoreit. Scalingautoregressivevideomodels. In ICLR,2019. BichenWu, ChenfengXu, XiaoliangDai, AlvinWan, PeizhaoZhang, MasayoshiTomizuka, Kurt Keutzer,andPeterVajda. Visualtransformers: Token-basedimagerepresentationandprocessing forcomputervision. arxiv,2020. YuxinWuandKaimingHe. Groupnormalization. InECCV,2018. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improvesimagenetclassification. InCVPR,2020. XiaohuaZhai,AvitalOliver,AlexanderKolesnikov,andLucasBeyer. S4L:Self-SupervisedSemi- SupervisedLearning. InICCV,2019a. Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprintarXiv:1910.04867,2019b. HengshuangZhao,JiayaJia,andVladlenKoltun. Exploringself-attentionforimagerecognition. In CVPR,2020. 12 PublishedasaconferencepaperatICLR2021 Models Dataset Epochs BaseLR LRdecay Weightdecay Dropout ViT-B/{16,32} JFT-300M 7 8·10−4 linear 0.1 0.0 ViT-L/32 JFT-300M 7 6·10−4 linear 0.1 0.0 ViT-L/16 JFT-300M 7/14 4·10−4 linear 0.1 0.0 ViT-H/14 JFT-300M 14 3·10−4 linear 0.1 0.0 R50x{1,2} JFT-300M 7 10−3 linear 0.1 0.0 R101x1 JFT-300M 7 8·10−4 linear 0.1 0.0 R152x{1,2} JFT-300M 7 6·10−4 linear 0.1 0.0 R50+ViT-B/{16,32} JFT-300M 7 8·10−4 linear 0.1 0.0 R50+ViT-L/32 JFT-300M 7 2·10−4 linear 0.1 0.0 R50+ViT-L/16 JFT-300M 7/14 4·10−4 linear 0.1 0.0 ViT-B/{16,32} ImageNet-21k 90 10−3 linear 0.03 0.1 ViT-L/{16,32} ImageNet-21k 30/90 10−3 linear 0.03 0.1 ViT-∗ ImageNet 300 3·10−3 cosine 0.3 0.1 Table3: Hyperparametersfortraining. Allmodelsaretrainedwithabatchsizeof4096andlearn- ing rate warmup of 10k steps. For ImageNet we found it beneficial to additionally apply gradient clippingatglobalnorm1. Trainingresolutionis224. APPENDIX A MULTIHEAD SELF-ATTENTION Standardqkvself-attention(SA,Vaswanietal.(2017))isapopularbuildingblockforneuralarchi- tectures. Foreachelementinaninputsequencez ∈ RN×D, wecomputeaweightedsumoverall values v in the sequence. The attention weights A are based on the pairwise similarity between ij twoelementsofthesequenceandtheirrespectivequeryqiandkeykj representations. [q,k,v]=zU qkv U qkv ∈RD×3Dh, (5) (cid:16) (cid:112) (cid:17) A=softmax qk(cid:62)/ D A∈RN×N, (6) h SA(z)=Av. (7) Multiheadself-attention(MSA)isanextensionofSAinwhichwerunk self-attentionoperations, called“heads”,inparallel,andprojecttheirconcatenatedoutputs. Tokeepcomputeandnumberof parametersconstantwhenchangingk,D (Eq.5)istypicallysettoD/k. h MSA(z)=[SA 1(z);SA 2(z);··· ;SA k(z)]U msa U msa ∈Rk·Dh×D (8) B EXPERIMENT DETAILS B.1 TRAINING Table 3 summarizes our training setups for our different models. We found strong regularization to be key when training models from scratch on ImageNet. Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings. HybridmodelsaretrainedwiththeexactsetupastheirViTcounterparts. Finally,all trainingisdoneonresolution224. B.1.1 FINE-TUNING Wefine-tuneallViTmodelsusingSGDwithamomentumof0.9. Werunasmallgridsearchover learningrates,seelearningraterangesinTable4.Todoso,weusesmallsub-splitsfromthetraining set(10%forPetsandFlowers,2%forCIFAR,1%ImageNet)asdevelopmentsetandtrainonthe remaining data. For final results we train on the entire training set and evaluate on the respective test data. For fine-tuning ResNets and hybrid models we use the exact same setup, with the only exceptionofImageNetwhereweaddanothervalue0.06tothelearningratesweep. Additionally, 13 PublishedasaconferencepaperatICLR2021 Dataset Steps BaseLR ImageNet 20000 {0.003,0.01,0.03,0.06} CIFAR100 10000 {0.001,0.003,0.01,0.03} CIFAR10 10000 {0.001,0.003,0.01,0.03} Oxford-IIITPets 500 {0.001,0.003,0.01,0.03} OxfordFlowers-102 500 {0.001,0.003,0.01,0.03} VTAB(19tasks) 2500 0.01 Table4:Hyperparametersforfine-tuning.Allmodelsarefine-tunedwithcosinelearningratedecay, abatchsizeof512,noweightdecay,andgradclippingatglobalnorm1.Ifnotmentionedotherwise, fine-tuningresolutionis384. for ResNets we also run the setup of Kolesnikov et al. (2020) and select the best results across thisrunandoursweep. Finally,ifnotmentionedotherwise,allfine-tuningexperimentsrunat384 resolution(runningfine-tuningatdifferentresolutionthantrainingiscommonpractice(Kolesnikov etal.,2020)). WhentransferringViTmodelstoanotherdataset,weremovethewholehead(twolinearlayers)and replaceitbyasingle,zero-initializedlinearlayeroutputtingthenumberofclassesrequiredbythe targetdataset. Wefoundthistobealittlemorerobustthansimplyre-initializingtheverylastlayer. For VTAB we follow the protocol in Kolesnikov et al. (2020), and use the same hyperparameter settingforalltasks. Weusealearningrateof0.01andtrainfor2500steps(Tab.4). Wechosethis settingbyrunningasmallsweepovertwolearningratesandtwoschedules,andselectingthesetting withthehighestVTABscoreonthe200-examplevalidationsets.Wefollowthepre-processingused inKolesnikovetal.(2020),exceptthatwedonotusetask-specificinputresolutions.Insteadwefind thatVisionTransformerbenefitsmostfromahighresolution(384×384)foralltasks. B.1.2 SELF-SUPERVISION Weemploythemaskedpatchpredictionobjectiveforpreliminaryself-supervisionexperiments. To do so we corrupt 50% of patch embeddings by either replacing their embeddings with a learnable [mask] embedding (80%), a random other patch embedding (10%) or just keeping them as is (10%). ThissetupisverysimilartotheoneusedforlanguagebyDevlinetal.(2019). Finally,we predictthe3-bit,meancolor(i.e.,512colorsintotal)ofeverycorruptedpatchusingtheirrespective patchrepresentations. Wetrainedourself-supervisedmodelfor1Msteps(ca. 14epochs)withbatchsize4096onJFT.We useAdam,withabaselearningrateof2·10−4,warmupof10kstepsandcosinelearningratedecay. As prediction targets for pretraining we tried the following settings: 1) predicting only the mean, 3bitcolor(i.e.,1predictionof512colors),2)predictinga4×4downsizedversionofthe16×16 patchwith3bitcolorsinparallel(i.e.,16predictionsof512colors),3)regressiononthefullpatch usingL2(i.e.,256regressionsonthe3RGBchannels). Surprisingly,wefoundthatallworkedquite well,thoughL2wasslightlyworse. Wereportfinalresultsonlyforoption1)becauseithasshown bestfew-shotperformance. Wealsoexperimentedwith15%corruptionrateasusedbyDevlinetal.
4. (2019)butresultswerealsoslightlyworseonourfew-shotmetrics. Lastly, we would like to remark that our instantiation of masked patch prediction doesn’t require such an enormous amount of pretraining nor a large dataset such as JFT in order to lead to sim- ilar performance gains on ImageNet classification. That is, we observed diminishing returns on downstream performance after 100k pretraining steps, and see similar gains when pretraining on ImageNet. C ADDITIONAL RESULTS Wereportdetailedresultscorrespondingtothefigurespresentedinthepaper. Table5corresponds to Figure 3 from the paper and shows transfer performance of different ViT models pre-trained on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. Table 6 corresponds to 14 PublishedasaconferencepaperatICLR2021 ViT-B/16 ViT-B/32 ViT-L/16 ViT-L/32 ViT-H/14 ImageNet CIFAR-10 98.13 97.77 97.86 97.94 - CIFAR-100 87.13 86.31 86.35 87.07 - ImageNet 77.91 73.38 76.53 71.16 - ImageNetReaL 83.57 79.56 82.19 77.83 - OxfordFlowers-102 89.49 85.43 89.66 86.36 - Oxford-IIIT-Pets 93.81 92.04 93.64 91.35 - ImageNet-21k CIFAR-10 98.95 98.79 99.16 99.13 99.27 CIFAR-100 91.67 91.97 93.44 93.04 93.82 ImageNet 83.97 81.28 85.15 80.99 85.13 ImageNetReaL 88.35 86.63 88.40 85.65 88.70 OxfordFlowers-102 99.38 99.11 99.61 99.19 99.51 Oxford-IIIT-Pets 94.43 93.02 94.73 93.09 94.82 JFT-300M CIFAR-10 99.00 98.61 99.38 99.19 99.50 CIFAR-100 91.87 90.49 94.04 92.52 94.55 ImageNet 84.15 80.73 87.12 84.37 88.04 ImageNetReaL 88.85 86.27 89.99 88.28 90.33 OxfordFlowers-102 99.56 99.27 99.56 99.45 99.68 Oxford-IIIT-Pets 95.80 93.40 97.11 95.83 97.56 Table5: Top1accuracy(in%)ofVisionTransformeronvariousdatasetswhenpre-trainedonIm- ageNet,ImageNet-21korJFT300M.ThesevaluescorrespondtoFigure3inthemaintext. Models are fine-tuned at 384 resolution. Note that the ImageNet results are computed without additional techniques(Polyakaveragingand512resolutionimages)usedtoachieveresultsinTable2. Epochs ImageNet ImageNetReaL CIFAR-10 CIFAR-100 Pets Flowers exaFLOPs name ViT-B/32 7 80.73 86.27 98.61 90.49 93.40 99.27 55 ViT-B/16 7 84.15 88.85 99.00 91.87 95.80 99.56 224 ViT-L/32 7 84.37 88.28 99.19 92.52 95.83 99.45 196 ViT-L/16 7 86.30 89.43 99.38 93.46 96.81 99.66 783 ViT-L/16 14 87.12 89.99 99.38 94.04 97.11 99.56 1567 ViT-H/14 14 88.08 90.36 99.50 94.71 97.11 99.71 4262 ResNet50x1 7 77.54 84.56 97.67 86.07 91.11 94.26 50 ResNet50x2 7 82.12 87.94 98.29 89.20 93.43 97.02 199 ResNet101x1 7 80.67 87.07 98.48 89.17 94.08 95.95 96 ResNet152x1 7 81.88 87.96 98.82 90.22 94.17 96.94 141 ResNet152x2 7 84.97 89.69 99.06 92.05 95.37 98.62 563 ResNet152x2 14 85.56 89.89 99.24 91.92 95.75 98.75 1126 ResNet200x3 14 87.22 90.15 99.34 93.53 96.32 99.04 3306 R50x1+ViT-B/32 7 84.90 89.15 99.01 92.24 95.75 99.46 106 R50x1+ViT-B/16 7 85.58 89.65 99.14 92.63 96.65 99.40 274 R50x1+ViT-L/32 7 85.68 89.04 99.24 92.93 96.97 99.43 246 R50x1+ViT-L/16 7 86.60 89.72 99.18 93.64 97.03 99.40 859 R50x1+ViT-L/16 14 87.12 89.76 99.31 93.89 97.36 99.11 1668 Table6: Detailedresultsofmodelscalingexperiments. ThesecorrespondtoFigure5inthemain paper. We show transfer accuracy on several datasets, as well as the pre-training compute (in ex- aFLOPs). Figure5fromthepaperandshowsthetransferperformanceofViT,ResNet,andhybridmodelsof varyingsize,aswellastheestimatedcomputationalcostoftheirpre-training. D ADDITIONAL ANALYSES D.1 SGDVS. ADAMFORRESNETS ResNetsaretypicallytrainedwithSGDandouruseofAdamasoptimizerisquiteunconventional. Here we show the experiments that motivated this choice. Namely, we compare the fine-tuning 15 PublishedasaconferencepaperatICLR2021 ResNet50 ResNet152x2 Dataset Adam SGD Adam SGD ImageNet 77.54 78.24 84.97 84.37 CIFAR10 97.67 97.46 99.06 99.07 CIFAR100 86.07 85.17 92.05 91.06 Oxford-IIITPets 91.11 91.00 95.37 94.79 OxfordFlowers-102 94.26 92.06 98.62 99.32 Average 89.33 88.79 94.01 93.72 Table7: Fine-tuningResNetmodelspre-trainedwithAdamandSGD.
5. 0.8
6. 0.6
7. 0.7 0.5
8. 0.4 Models 0.6 Models All All
9. 0.3 Depth Depth Patch size 0.5 Patch size
10. 0.2 Width MLP Width MLP Width Width
11. 0.4 100 101 100 101 Relative Compute Relative Compute Figure8: ScalingdifferentmodeldimensionsoftheVisionTransformer. performance of two ResNets – 50x1 and 152x2 – pre-trained on JFT with SGD and Adam. For SGD,weusethehyperparametersrecommendedbyKolesnikovetal.(2020). Resultsarepresented in Table 7. Adam pre-training outperforms SGD pre-training on most datasets and on average. ThisjustifiesthechoiceofAdamastheoptimizerusedtopre-trainResNetsonJFT.Notethatthe absolutenumbersarelowerthanthosereportedbyKolesnikovetal.(2020),sincewepre-trainonly for7epochs,not30. D.2 TRANSFORMERSHAPE WeranablationsonscalingdifferentdimensionsoftheTransformerarchitecturetofindoutwhich are best suited for scaling to very large models. Figure 8 shows 5-shot performance on ImageNet fordifferentconfigurations. AllconfigurationsarebasedonaViTmodelwith8layers,D = 1024, D = 2048 and a patch size of 32, the intersection of all lines. We can see that scaling the MLP depth results in the biggest improvements which are clearly visible up until 64 layers. However, diminishing returns are already visible after 16 layers. Interestingly, scaling the width of the net- work seems to result in the smallest changes. Decreasing the patch size and thus increasing the effectivesequencelengthshowssurprisinglyrobustimprovementswithoutintroducingparameters. Thesefindingssuggestthatcomputemightbeabetterpredictorofperformancethanthenumberof parameters,andthatscalingshouldemphasizedepthoverwidthifany. Overall,wefindthatscaling alldimensionsproportionallyresultsinrobustimprovements. D.3 HEADTYPEAND CLASS TOKEN InordertostayascloseaspossibletotheoriginalTransformermodel,wemadeuseofanadditional [class] token, which is taken as image representation. The output of this token is then trans- formedintoaclasspredictionviaasmallmulti-layerperceptron(MLP)withtanhasnon-linearity inthesinglehiddenlayer. This design is inherited from the Transformer model for text, and we use it throughout the main paper. An initial attempt at using only image-patch embeddings, globally average-pooling (GAP) them,followedbyalinearclassifier—justlikeResNet’sfinalfeaturemap—performedverypoorly. However, we found that this is neither due to the extra token, nor to the GAP operation. Instead, 16 tohs5 teNegamI tohs5 egarevA PublishedasaconferencepaperatICLR2021 60 CLS-Token, lr=8e-4 55 GAP, lr=8e-4 GAP, lr=3e-4 50 45 40 35 30 25 0 1 2 3 4 5 6 7 Epochs of training Figure 9: Comparison of class-token and global average pooling classifiers. Both work similarly well,butrequiredifferentlearning-rates. Pos. Emb. Default/Stem EveryLayer EveryLayer-Shared NoPos. Emb. 0.61382 N/A N/A 1-DPos. Emb. 0.64206 0.63964 0.64292 2-DPos. Emb. 0.64001 0.64046 0.64022 Rel. Pos. Emb. 0.64032 N/A N/A Table8: ResultsoftheablationstudyonpositionalembeddingswithViT-B/16modelevaluatedon ImageNet5-shotlinear. thedifferenceinperformanceisfullyexplainedbytherequirementforadifferentlearning-rate,see Figure9. D.4 POSITIONALEMBEDDING Weranablationsondifferentwaysofencodingspatialinformationusingpositionalembedding. We triedthefollowingcases: • Providingnopositionalinformation: Consideringtheinputsasabagofpatches. • 1-dimensional positional embedding: Considering the inputs as a sequence of patches in therasterorder(defaultacrossallotherexperimentsinthispaper). • 2-dimensional positional embedding: Considering the inputs as a grid of patches in two dimensions. In this case, two sets of embeddings are learned, each for one of the axes, X-embedding,andY-embedding,eachwithsizeD/2. Then,basedonthecoordinateon the path in the input, we concatenate the X and Y embedding to get the final positional embeddingforthatpatch. • Relativepositionalembeddings: Consideringtherelativedistancebetweenpatchestoen- code the spatial information as instead of their absolute position. To do so, we use 1- dimensionalRelativeAttention,inwhichwedefinetherelativedistanceallpossiblepairs ofpatches. Thus, foreverygivenpair(oneasquery, andtheotheraskey/valueintheat- tention mechanism), we have an offset p −p , where each offset is associated with an q k embedding. Then, we simply run extra attention, where we use the original query (the content of query), but use relative positional embeddings as keys. We then use the log- its from the relative attention as a bias term and add it to the logits of the main attention (content-basedattention)beforeapplyingthesoftmax. In addition to different ways of encoding spatial information, we also tried different ways of in- corporating this information in our model. For the 1-dimensional and 2-dimensional positional embeddings, wetriedthreedifferentcases: (1)addpositionalembeddingstotheinputsrightafter 17 ]%[ ycarucca tohs-5 raenil teNegamI PublishedasaconferencepaperatICLR2021 ViT-L16 ViT-L16 ViT-L16 7 epochs, LR=0.0002, WD=0.01 7 epochs, LR=0.0004, WD=0.1 14 epochs, LR=0.0004, WD=0.1 1 1 1 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 1 2 3 4 5 6 7 8 9 1011121314 1 1 2 3 4 5 6 7 8 9 1011121314 1 1 2 3 4 5 6 7 8 9 1011121314 1 Input patch column Input patch column Input patch column Figure10: Positionembeddingsofmodelstrainedwithdifferenthyperparameters. the stem of them model and before feeding the inputs to the Transformer encoder (default across all other experiments in this paper); (2) learn and add positional embeddings to the inputs at the beginningofeachlayer; (3)addalearnedpositionalembeddingstotheinputsatthebeginningof eachlayer(sharedbetweenlayers). Table8summarizestheresultsfromthisablationstudyonaViT-B/16model. Aswecansee,while thereisalargegapbetweentheperformancesofthemodelwithnopositionalembeddingandmod- els with positional embedding, there is little to no difference between different ways of encoding positional information. We speculate that since our Transformer encoder operates on patch-level inputs,asopposedtopixel-level,thedifferencesinhowtoencodespatialinformationislessimpor- tant. Moreprecisely,inpatch-levelinputs,thespatialdimensionsaremuchsmallerthantheoriginal pixel-levelinputs, e.g., 14×14asopposedto224×224, andlearningtorepresentthespatialre- lationsinthisresolutionisequallyeasyforthesedifferentpositionalencodingstrategies. Evenso, thespecificpatternofpositionembeddingsimilaritylearnedbythenetworkdependsonthetraining hyperparameters(Figure10). ViT-L/16 R50x1 + ViT-L/16 120 120 100 100 80 80 60 60 40 40 Head 1 Head 1 20 Head 2 20 Head 2 Head 3 Head 3 ... ... 0 0 0 5 10 15 20 0 5 10 15 20 Network depth (layer) Network depth (layer) Figure11: Sizeofattendedareabyheadandnetworkdepth. Attentiondistancewascomputedfor 128exampleimagesbyaveragingthedistancebetweenthequerypixelandallotherpixels,weighted by the attention weight. Each dot shows the mean attention distance across images for one of 16 headsatonelayer. Imagewidthis224pixels. D.5 EMPIRICALCOMPUTATIONALCOSTS Wearealsointerestedinreal-worldspeedofthearchitecturesonourhardware,whichisnotalways wellpredictedbytheoreticalFLOPsduetodetailslikelanewidthsandcachesizes.Forthispurpose, 18 wor hctap tupnI )slexip( ecnatsid noitnetta naeM ywtiorra lhimctiasp e tnuisponCI ywtiorra lhimctiasp e tnuisponCI ytiralimis enisoC PublishedasaconferencepaperatICLR2021 weperformtimingofinferencespeedforthemainmodelsofinterest,onaTPUv3accelerator;the differencebetweeninferenceandbackpropspeedisaconstantmodel-independentfactor. Figure12(left)showshowmanyimagesonecorecanhandlepersecond,acrossvariousinputsizes. Everysinglepointreferstothepeakperformancemeasuredacrossawiderangeofbatch-sizes. As canbeseen,thetheoreticalbi-quadraticscalingofViTwithimagesizeonlybarelystartshappening forthelargestmodelsatthelargestresolutions. Another quantity of interest is the largest batch-size each model can fit onto a core, larger being betterforscalingtolargedatasets. Figure12(right)showsthisquantityforthesamesetofmodels. ThisshowsthatlargeViTmodelshaveaclearadvantageintermsofmemory-efficiencyoverResNet models. R50x1 ViT-B/32 ViT-B/16 ViT-H/14 R50x2 ViT-L/32 ViT-L/16 R152x4 103 104 103 102 102 64 128 224 384 512 64 128 224 384 512 Input size [px] Input size [px] Figure 12: Left: Real wall-clock timings of various architectures across input sizes. ViT models havespeedcomparabletosimilarResNets. Right:Largestper-corebatch-sizefittingondevicewith variousarchitecturesacrossinputsizes. ViTmodelsareclearlymorememory-efficient. D.6 AXIALATTENTION AxialAttention(Huangetal.,2020;Hoetal.,2019)isasimple,yeteffectivetechniquetorunself- attentiononlargeinputsthatareorganizedasmultidimensionaltensors. Thegeneralideaofaxial attention is to perform multiple attention operations, each along a single axis of the input tensor, insteadofapplying1-dimensionalattentiontotheflattenedversionoftheinput. Inaxialattention, eachattentionmixesinformationalongaparticularaxis,whilekeepinginformationalongtheother axesindependent. Alongthisline,Wangetal.(2020b)proposedtheAxialResNetmodelinwhich all the convolutions with kernel size 3×3 in a ResNet50 are replaced by axial self-attention, i.e. a row and column attention, augmented by relative positional encoding. We have implemented AxialResNetasabaselinemodel.3. Moreover, we have modified ViT to process inputs in the 2-dimensional shape, instead of a 1- dimensional sequence of patches, and incorporate Axial Transformer blocks, in which instead of a self-attention followed by an MLP, we have a a row-self-attention plus an MLP followed by a column-self-attentionplusanMLP. Figure13,presenttheperformanceofAxialResNet,Axial-ViT-B/32andAxial-ViT-B/16onIma- geNet5shotlinear,whenpretrainedonJFTdataset,versesthepretrainingcompute,bothintermsof numberofFLOPsandinferencetime(exampleperseconds). Aswecansee,bothAxial-ViT-B/32 andAxial-ViT-B/16dobetterthantheirViT-Bcounterpartintermsofperformance,butitcomesat 3Ourimplementationisbasedontheopen-sourcedPyTorchimplementationinhttps://github.com/ csrhddlam/axial-deeplab. In our experiments, we reproduced the scores reported in (Wang et al., 2020b)intermsofaccuracy,however,ourimplementation,similartotheopen-sourceimplementation,isvery slow on TPUs. Therefore, we were not able to use it for extensive large-scale experiments. These may be unlockedbyacarefullyoptimizedimplementation. 19 ]eroc/ces/gmi[ deeps ecnerefni kaeP ezis-hctab eroc-rep tsegraL PublishedasaconferencepaperatICLR2021 AxialViT-B/16 AxialViT-B/16
12. 0.650 0.650 ViT-B/16 ViT-B/16
13. 0.625 0.625
14. 0.600 0.600 AxialViT-B/32 AxialViT-B/32
15. 0.575 0.575 ViT-B/32 ViT-B/32 AxialResNet50 AxialResNet50
16. 0.550 0.550
17. 0.525 0.525
18. 0.500 ResNet50 0.500 ResNet50 102 103 102 Total compute [exaFLOPs] Peak inference speed [img/sec/core] Figure13: PerformanceofAxial-Attentionbasedmodels,intermsoftop-1accuracyonImageNet 5-shotlinear,versustheirspeedintermsofnumberofFLOPs(left)andinferencetime(left). thecostofmorecompute.ThisisbecauseinAxial-ViTmodels,eachTransformerblockwithglobal self-attentionisreplacedbytwoAxialTransformerblocks,onewithrowandonewithcolumnself- attention and although the sequence length that self-attention operates on is smaller in axial case, there is a extra MLP per Axial-ViT block. For the AxialResNet, although it looks reasonable in terms of accuracy/compute trade-off (Figure 13, left), the naive implementation is extremely slow onTPUs(Figure13,right). D.7 ATTENTIONDISTANCE To understand how ViT uses self-attention to integrate information across the image, we analyzed the average distance spanned by attention weights at different layers (Figure 11). This “attention distance”isanalogoustoreceptivefieldsizeinCNNs. Averageattentiondistanceishighlyvariable acrossheadsinlowerlayers,withsomeheadsattendingtomuchoftheimage,whileothersattend tosmallregionsatornearthequerylocation. Asdepthincreases,attentiondistanceincreasesforall heads. Inthesecondhalfofthenetwork,mostheadsattendwidelyacrosstokens. D.8 ATTENTIONMAPS Tocomputemapsoftheattentionfromtheoutputtokentotheinputspace(Figures6and14), we used Attention Rollout (Abnar & Zuidema, 2020). Briefly, we averaged attention weights of ViT- L/16acrossallheadsandthenrecursivelymultipliedtheweightmatricesofalllayers.Thisaccounts forthemixingofattentionacrosstokensthroughalllayers. D.9 OBJECTNETRESULTS WealsoevaluateourflagshipViT-H/14modelontheObjectNetbenchmarkfollowingtheevaluation setupinKolesnikovetal.(2020),resultingin82.1%top-5accuracyand61.7%top-1accuracy. D.10 VTABBREAKDOWN Table9showsthescoresattainedoneachoftheVTAB-1ktasks. 20 ycarucca 1-pot raenil tohs-5 teNegamI ycarucca 1-pot raenil tohs-5 teNegamI PublishedasaconferencepaperatICLR2021 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 Figure14: FurtherexampleattentionmapsasinFigure6(randomselection). 21 PublishedasaconferencepaperatICLR2021 Table9: BreakdownofVTAB-1kperformanceacrosstasks. ViT-H/14(JFT) 95.3 85.5 75.2 99.7 97.2 65.0 88.9 83.3 96.7 91.4 76.6 91.7 63.8 53.1 79.4 63.3 84.5 33.2 51.2 77.6 ViT-L/16(JFT) 95.4 81.9 74.3 99.7 96.7 63.5 87.4 83.6 96.5 89.7 77.1 86.4 63.1 49.7 74.5 60.5 82.2 36.2 51.1 76.3 ViT-L/16(I21k)90.8 84.1 74.1 99.3 92.7 61.0 80.9 82.5 95.6 85.2 75.3 70.3 56.1 41.9 74.7 64.9 79.9 30.5 41.7 72.7 22 101hcetlaC 001-RAFIC DTD 201srewolF steP 793nuS NHVS noylemaC TASoruE 54csiseR yhtaponiteR tnuoC-rvelC tsiD-rvelC baLMD coL-rpSd irO-rpSd tsiD-ITTIK mizA-BRONs velE-BRONs naeM

---
*Processed on 2025-08-07 18:35:56 UTC*
*Processing time: 96.04 seconds*