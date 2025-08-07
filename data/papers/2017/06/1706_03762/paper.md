# Attention Is All You Need

**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**arXiv ID**: 1706.03762
**Published**: 2017-06-12
**Categories**: cs.CL, cs.LG
**Keywords**: cs.CL, cs.LG, model, training, data, attention, transformer

## Abstract Summary
**Abstract Summary**: The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks in an encoder-decoder configuration. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more ...

## Original Abstract
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks in an encoder-decoder configuration. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014
English-to-German translation task, improving over the existing best results,
including ensembles by over 2 BLEU. On the WMT 2014 English-to-French
translation task, our model establishes a new single-model state-of-the-art
BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction
of the training costs of the best models from the literature. We show that the
Transformer generalizes well to other tasks by applying it successfully to
English constituency parsing both with large and limited training data.

## Header

Providedproperattributionisprovided,Googleherebygrantspermissionto
reproducethetablesandfiguresinthispapersolelyforuseinjournalisticor
scholarlyworks.
Attention Is All You Need
AshishVaswani∗ NoamShazeer∗ NikiParmar∗ JakobUszkoreit∗
GoogleBrain GoogleBrain GoogleResearch GoogleResearch
avaswani@google.com noam@google.com nikip@google.com usz@google.com
LlionJones∗ AidanN.Gomez∗ † ŁukaszKaiser∗
GoogleResearch UniversityofToronto GoogleBrain
llion@google.com aidan@cs.toronto.edu lukaszkaiser@google.com
IlliaPolosukhin∗ ‡
illia.polosukhin@gmail.com

## Abstract

Thedominantsequencetransductionmodelsarebasedoncomplexrecurrentor
convolutionalneuralnetworksthatincludeanencoderandadecoder. Thebest
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
basedsolelyonattentionmechanisms,dispensingwithrecurrenceandconvolutions
entirely. Experiments on two machine translation tasks show these models to
besuperiorinqualitywhilebeingmoreparallelizableandrequiringsignificantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles,byover2BLEU.OntheWMT2014English-to-Frenchtranslationtask,
ourmodelestablishesanewsingle-modelstate-of-the-artBLEUscoreof41.8after
trainingfor3.5daysoneightGPUs,asmallfractionofthetrainingcostsofthe
bestmodelsfromtheliterature. WeshowthattheTransformergeneralizeswellto
othertasksbyapplyingitsuccessfullytoEnglishconstituencyparsingbothwith
largeandlimitedtrainingdata.
∗Equalcontribution.Listingorderisrandom.JakobproposedreplacingRNNswithself-attentionandstarted
theefforttoevaluatethisidea.Ashish,withIllia,designedandimplementedthefirstTransformermodelsand
hasbeencruciallyinvolvedineveryaspectofthiswork.Noamproposedscaleddot-productattention,multi-head
attentionandtheparameter-freepositionrepresentationandbecametheotherpersoninvolvedinnearlyevery
detail.Nikidesigned,implemented,tunedandevaluatedcountlessmodelvariantsinouroriginalcodebaseand
tensor2tensor.Llionalsoexperimentedwithnovelmodelvariants,wasresponsibleforourinitialcodebase,and
efficientinferenceandvisualizations.LukaszandAidanspentcountlesslongdaysdesigningvariouspartsofand
implementingtensor2tensor,replacingourearliercodebase,greatlyimprovingresultsandmassivelyaccelerating
ourresearch.
†WorkperformedwhileatGoogleBrain.
‡WorkperformedwhileatGoogleResearch.
31stConferenceonNeuralInformationProcessingSystems(NIPS2017),LongBeach,CA,USA.
3202
guA
]LC.sc[
7v26730.6071:viXra

## Introduction

Recurrentneuralnetworks,longshort-termmemory[13]andgatedrecurrent[7]neuralnetworks
inparticular,havebeenfirmlyestablishedasstateoftheartapproachesinsequencemodelingand
transductionproblemssuchaslanguagemodelingandmachinetranslation[35,2,5]. Numerous
effortshavesincecontinuedtopushtheboundariesofrecurrentlanguagemodelsandencoder-decoder
architectures[38,24,15].
Recurrentmodelstypicallyfactorcomputationalongthesymbolpositionsoftheinputandoutput
sequences. Aligningthepositionstostepsincomputationtime,theygenerateasequenceofhidden
statesh ,asafunctionoftheprevioushiddenstateh andtheinputforpositiont. Thisinherently
t t−1
sequentialnatureprecludesparallelizationwithintrainingexamples,whichbecomescriticalatlonger
sequencelengths,asmemoryconstraintslimitbatchingacrossexamples. Recentworkhasachieved
significantimprovementsincomputationalefficiencythroughfactorizationtricks[21]andconditional
computation[32],whilealsoimprovingmodelperformanceincaseofthelatter. Thefundamental
constraintofsequentialcomputation,however,remains.
Attentionmechanismshavebecomeanintegralpartofcompellingsequencemodelingandtransduc-
tionmodelsinvarioustasks,allowingmodelingofdependencieswithoutregardtotheirdistancein
theinputoroutputsequences[2,19]. Inallbutafewcases[27],however,suchattentionmechanisms
areusedinconjunctionwitharecurrentnetwork.
InthisworkweproposetheTransformer,amodelarchitectureeschewingrecurrenceandinstead
relyingentirelyonanattentionmechanismtodrawglobaldependenciesbetweeninputandoutput.
TheTransformerallowsforsignificantlymoreparallelizationandcanreachanewstateoftheartin
translationqualityafterbeingtrainedforaslittleastwelvehoursoneightP100GPUs.

## 2 Background

ThegoalofreducingsequentialcomputationalsoformsthefoundationoftheExtendedNeuralGPU
[16],ByteNet[18]andConvS2S[9],allofwhichuseconvolutionalneuralnetworksasbasicbuilding
block,computinghiddenrepresentationsinparallelforallinputandoutputpositions.Inthesemodels,
thenumberofoperationsrequiredtorelatesignalsfromtwoarbitraryinputoroutputpositionsgrows
inthedistancebetweenpositions,linearlyforConvS2SandlogarithmicallyforByteNet. Thismakes
it more difficult to learn dependencies between distant positions [12]. In the Transformer this is
reducedtoaconstantnumberofoperations, albeitatthecostofreducedeffectiveresolutiondue
to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
describedinsection3.2.
Self-attention,sometimescalledintra-attentionisanattentionmechanismrelatingdifferentpositions
ofasinglesequenceinordertocomputearepresentationofthesequence. Self-attentionhasbeen
usedsuccessfullyinavarietyoftasksincludingreadingcomprehension,abstractivesummarization,
textualentailmentandlearningtask-independentsentencerepresentations[4,27,28,22].
End-to-endmemorynetworksarebasedonarecurrentattentionmechanisminsteadofsequence-
alignedrecurrenceandhavebeenshowntoperformwellonsimple-languagequestionansweringand
languagemodelingtasks[34].
To the best of our knowledge, however, the Transformer is the first transduction model relying
entirelyonself-attentiontocomputerepresentationsofitsinputandoutputwithoutusingsequence-
alignedRNNsorconvolution. Inthefollowingsections,wewilldescribetheTransformer,motivate
self-attentionanddiscussitsadvantagesovermodelssuchas[17,18]and[9].

## 3 ModelArchitecture

Mostcompetitiveneuralsequencetransductionmodelshaveanencoder-decoderstructure[5,2,35].
Here, the encoder maps an input sequence of symbol representations (x ,...,x ) to a sequence
1 n
of continuous representations z = (z ,...,z ). Given z, the decoder then generates an output
1 n
sequence(y ,...,y )ofsymbolsoneelementatatime. Ateachstepthemodelisauto-regressive
1 m
[10],consumingthepreviouslygeneratedsymbolsasadditionalinputwhengeneratingthenext.
Figure1: TheTransformer-modelarchitecture.
TheTransformerfollowsthisoverallarchitectureusingstackedself-attentionandpoint-wise,fully
connectedlayersforboththeencoderanddecoder,shownintheleftandrighthalvesofFigure1,
respectively.
3.1 EncoderandDecoderStacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. Thefirstisamulti-headself-attentionmechanism,andthesecondisasimple,position-
wisefullyconnectedfeed-forwardnetwork. Weemployaresidualconnection[11]aroundeachof
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x+Sublayer(x)),whereSublayer(x)isthefunctionimplementedbythesub-layer
itself. Tofacilitatetheseresidualconnections,allsub-layersinthemodel,aswellastheembedding
layers,produceoutputsofdimensiond =512.
model
Decoder: ThedecoderisalsocomposedofastackofN =6identicallayers. Inadditiontothetwo
sub-layersineachencoderlayer,thedecoderinsertsathirdsub-layer,whichperformsmulti-head
attentionovertheoutputoftheencoderstack. Similartotheencoder,weemployresidualconnections
aroundeachofthesub-layers,followedbylayernormalization. Wealsomodifytheself-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking,combinedwithfactthattheoutputembeddingsareoffsetbyoneposition,ensuresthatthe
predictionsforpositionicandependonlyontheknownoutputsatpositionslessthani.
3.2 Attention
Anattentionfunctioncanbedescribedasmappingaqueryandasetofkey-valuepairstoanoutput,
wherethequery,keys,values,andoutputareallvectors. Theoutputiscomputedasaweightedsum
ScaledDot-ProductAttention Multi-HeadAttention
Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several
attentionlayersrunninginparallel.
ofthevalues,wheretheweightassignedtoeachvalueiscomputedbyacompatibilityfunctionofthe
querywiththecorrespondingkey.
3.2.1 ScaledDot-ProductAttention
Wecallourparticularattention"ScaledDot-ProductAttention"(Figure2). Theinputconsistsof
queriesandkeysofdimensiond k,a√ndvaluesofdimensiond v. Wecomputethedotproductsofthe
querywithallkeys,divideeachby d ,andapplyasoftmaxfunctiontoobtaintheweightsonthe
values.
Inpractice,wecomputetheattentionfunctiononasetofqueriessimultaneously,packedtogether
intoamatrixQ. ThekeysandvaluesarealsopackedtogetherintomatricesK andV. Wecompute
thematrixofoutputsas:
QKT
Attention(Q,K,V)=softmax( √ )V (1)
Thetwomostcommonlyusedattentionfunctionsareadditiveattention[2],anddot-product(multi-
plicative)attention. Dot-productattentionisidenticaltoouralgorithm,exceptforthescalingfactor
of √1 . Additiveattentioncomputesthecompatibilityfunctionusingafeed-forwardnetworkwith
asinglehiddenlayer. Whilethetwoaresimilarintheoreticalcomplexity,dot-productattentionis
muchfasterandmorespace-efficientinpractice,sinceitcanbeimplementedusinghighlyoptimized
matrixmultiplicationcode.
Whileforsmallvaluesofd thetwomechanismsperformsimilarly,additiveattentionoutperforms
dotproductattentionwithoutscalingforlargervaluesofd [3]. Wesuspectthatforlargevaluesof
d ,thedotproductsgrowlargeinmagnitude,pushingthesoftmaxfunctionintoregionswhereithas
extremelysmallgradients4. Tocounteractthiseffect,wescalethedotproductsby √1 .
3.2.2 Multi-HeadAttention
Insteadofperformingasingleattentionfunctionwithd -dimensionalkeys,valuesandqueries,
model
wefounditbeneficialtolinearlyprojectthequeries,keysandvalueshtimeswithdifferent,learned
linearprojectionstod ,d andd dimensions,respectively. Oneachoftheseprojectedversionsof
k k v
queries,keysandvalueswethenperformtheattentionfunctioninparallel,yieldingd -dimensional
4Toillustratewhythedotproductsgetlarge,assumethatthecomponentsofqandkareindependentrandom
variableswithmean0andvariance1.Thentheirdotproduct,q·k=(cid:80)dk
q k ,hasmean0andvarianced .
i=1 i i k
output values. These are concatenated and once again projected, resulting in the final values, as
depictedinFigure2.
Multi-headattentionallowsthemodeltojointlyattendtoinformationfromdifferentrepresentation
subspacesatdifferentpositions. Withasingleattentionhead,averaginginhibitsthis.
MultiHead(Q,K,V)=Concat(head ,...,head )WO
1 h
wherehead =Attention(QWQ,KWK,VWV)
i i i i
WheretheprojectionsareparametermatricesWQ ∈Rdmodel×dk,WK ∈Rdmodel×dk,WV ∈Rdmodel×dv
i i i
andWO ∈Rhdv×dmodel.
In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
d =d =d /h=64. Duetothereduceddimensionofeachhead,thetotalcomputationalcost
k v model
issimilartothatofsingle-headattentionwithfulldimensionality.
3.2.3 ApplicationsofAttentioninourModel
TheTransformerusesmulti-headattentioninthreedifferentways:
• In"encoder-decoderattention"layers,thequeriescomefromthepreviousdecoderlayer,
andthememorykeysandvaluescomefromtheoutputoftheencoder. Thisallowsevery
positioninthedecodertoattendoverallpositionsintheinputsequence. Thismimicsthe
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38,2,9].
• Theencodercontainsself-attentionlayers. Inaself-attentionlayerallofthekeys,values
andqueriescomefromthesameplace,inthiscase,theoutputofthepreviouslayerinthe
encoder. Eachpositionintheencodercanattendtoallpositionsinthepreviouslayerofthe
encoder.
• Similarly,self-attentionlayersinthedecoderalloweachpositioninthedecodertoattendto
allpositionsinthedecoderuptoandincludingthatposition. Weneedtopreventleftward
informationflowinthedecodertopreservetheauto-regressiveproperty. Weimplementthis
insideofscaleddot-productattentionbymaskingout(settingto−∞)allvaluesintheinput
ofthesoftmaxwhichcorrespondtoillegalconnections. SeeFigure2.
3.3 Position-wiseFeed-ForwardNetworks
Inadditiontoattentionsub-layers,eachofthelayersinourencoderanddecodercontainsafully
connectedfeed-forwardnetwork,whichisappliedtoeachpositionseparatelyandidentically. This
consistsoftwolineartransformationswithaReLUactivationinbetween.
FFN(x)=max(0,xW +b )W +b (2)
1 1 2 2
Whilethelineartransformationsarethesameacrossdifferentpositions,theyusedifferentparameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is d = 512, and the inner-layer has dimensionality
model
d =2048.
3.4 EmbeddingsandSoftmax
Similarlytoothersequencetransductionmodels,weuselearnedembeddingstoconverttheinput
tokensandoutputtokenstovectorsofdimensiond . Wealsousetheusuallearnedlineartransfor-
model
mationandsoftmaxfunctiontoconvertthedecoderoutputtopredictednext-tokenprobabilities. In
ourmodel,wesharethesameweightmatrixbetweenthetwoembeddinglayersandthepre-√softmax
lineartransformation,similarto[30]. Intheembeddinglayers,wemultiplythoseweightsby d .
model
Table1: Maximumpathlengths,per-layercomplexityandminimumnumberofsequentialoperations
fordifferentlayertypes. nisthesequencelength,distherepresentationdimension,kisthekernel
sizeofconvolutionsandrthesizeoftheneighborhoodinrestrictedself-attention.
LayerType ComplexityperLayer Sequential MaximumPathLength
Operations
Self-Attention O(n2·d) O(1) O(1)
Recurrent O(n·d2) O(n) O(n)
Convolutional O(k·n·d2) O(1) O(log (n))
Self-Attention(restricted) O(r·n·d) O(1) O(n/r)
3.5 PositionalEncoding
Sinceourmodelcontainsnorecurrenceandnoconvolution,inorderforthemodeltomakeuseofthe
orderofthesequence,wemustinjectsomeinformationabouttherelativeorabsolutepositionofthe
tokensinthesequence. Tothisend,weadd"positionalencodings"totheinputembeddingsatthe
bottomsoftheencoderanddecoderstacks. Thepositionalencodingshavethesamedimensiond
model
astheembeddings,sothatthetwocanbesummed. Therearemanychoicesofpositionalencodings,
learnedandfixed[9].
Inthiswork,weusesineandcosinefunctionsofdifferentfrequencies:
PE =sin(pos/100002i/dmodel)
(pos,2i)
PE =cos(pos/100002i/dmodel)
(pos,2i+1)
whereposisthepositionandiisthedimension. Thatis,eachdimensionofthepositionalencoding
correspondstoasinusoid. Thewavelengthsformageometricprogressionfrom2πto10000·2π. We
chosethisfunctionbecausewehypothesizeditwouldallowthemodeltoeasilylearntoattendby
relativepositions,sinceforanyfixedoffsetk,PE canberepresentedasalinearfunctionof
pos+k

## Pe .

pos
Wealsoexperimentedwithusinglearnedpositionalembeddings[9]instead,andfoundthatthetwo
versionsproducednearlyidenticalresults(seeTable3row(E)).Wechosethesinusoidalversion
becauseitmayallowthemodeltoextrapolatetosequencelengthslongerthantheonesencountered
duringtraining.
4 WhySelf-Attention
In this section we compare various aspects of self-attention layers to the recurrent and convolu-
tionallayerscommonlyusedformappingonevariable-lengthsequenceofsymbolrepresentations
(x ,...,x ) to another sequence of equal length (z ,...,z ), with x ,z ∈ Rd, such as a hidden
1 n 1 n i i
layerinatypicalsequencetransductionencoderordecoder. Motivatingouruseofself-attentionwe
considerthreedesiderata.
Oneisthetotalcomputationalcomplexityperlayer. Anotheristheamountofcomputationthatcan
beparallelized,asmeasuredbytheminimumnumberofsequentialoperationsrequired.
Thethirdisthepathlengthbetweenlong-rangedependenciesinthenetwork. Learninglong-range
dependenciesisakeychallengeinmanysequencetransductiontasks. Onekeyfactoraffectingthe
abilitytolearnsuchdependenciesisthelengthofthepathsforwardandbackwardsignalshaveto
traverseinthenetwork. Theshorterthesepathsbetweenanycombinationofpositionsintheinput
andoutputsequences,theeasieritistolearnlong-rangedependencies[12]. Hencewealsocompare
themaximumpathlengthbetweenanytwoinputandoutputpositionsinnetworkscomposedofthe
differentlayertypes.
AsnotedinTable1,aself-attentionlayerconnectsallpositionswithaconstantnumberofsequentially
executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of
computationalcomplexity,self-attentionlayersarefasterthanrecurrentlayerswhenthesequence
length n is smaller than the representation dimensionality d, which is most often the case with
sentencerepresentationsusedbystate-of-the-artmodelsinmachinetranslations,suchasword-piece
[38]andbyte-pair[31]representations. Toimprovecomputationalperformancefortasksinvolving
verylongsequences,self-attentioncouldberestrictedtoconsideringonlyaneighborhoodofsizerin
theinputsequencecenteredaroundtherespectiveoutputposition. Thiswouldincreasethemaximum
pathlengthtoO(n/r). Weplantoinvestigatethisapproachfurtherinfuturework.
Asingleconvolutionallayerwithkernelwidthk <ndoesnotconnectallpairsofinputandoutput
positions. DoingsorequiresastackofO(n/k)convolutionallayersinthecaseofcontiguouskernels,
orO(log (n))inthecaseofdilatedconvolutions[18], increasingthelengthofthelongestpaths
betweenanytwopositionsinthenetwork. Convolutionallayersaregenerallymoreexpensivethan
recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity
considerably, toO(k·n·d+n·d2). Evenwithk = n, however, thecomplexityofaseparable
convolutionisequaltothecombinationofaself-attentionlayerandapoint-wisefeed-forwardlayer,
theapproachwetakeinourmodel.
Assidebenefit,self-attentioncouldyieldmoreinterpretablemodels.Weinspectattentiondistributions
fromourmodelsandpresentanddiscussexamplesintheappendix. Notonlydoindividualattention
headsclearlylearntoperformdifferenttasks,manyappeartoexhibitbehaviorrelatedtothesyntactic
andsemanticstructureofthesentences.

## 5 Training

Thissectiondescribesthetrainingregimeforourmodels.
5.1 TrainingDataandBatching
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
sentencepairs. Sentenceswereencodedusingbyte-pairencoding[3],whichhasasharedsource-
targetvocabularyofabout37000tokens. ForEnglish-French,weusedthesignificantlylargerWMT
2014English-Frenchdatasetconsistingof36Msentencesandsplittokensintoa32000word-piece
vocabulary[38].Sentencepairswerebatchedtogetherbyapproximatesequencelength.Eachtraining
batchcontainedasetofsentencepairscontainingapproximately25000sourcetokensand25000
targettokens.
5.2 HardwareandSchedule
Wetrainedourmodelsononemachinewith8NVIDIAP100GPUs. Forourbasemodelsusing
thehyperparametersdescribedthroughoutthepaper,eachtrainingsteptookabout0.4seconds. We
trainedthebasemodelsforatotalof100,000stepsor12hours. Forourbigmodels,(describedonthe
bottomlineoftable3),steptimewas1.0seconds. Thebigmodelsweretrainedfor300,000steps
(3.5days).
5.3 Optimizer
WeusedtheAdamoptimizer[20]withβ =0.9,β =0.98andϵ=10−9. Wevariedthelearning
1 2
rateoverthecourseoftraining,accordingtotheformula:
lrate=d−0.5 ·min(step_num−0.5,step_num·warmup_steps−1.5) (3)
model
Thiscorrespondstoincreasingthelearningratelinearlyforthefirstwarmup_stepstrainingsteps,
anddecreasingitthereafterproportionallytotheinversesquarerootofthestepnumber. Weused
warmup_steps=4000.
5.4 Regularization
Weemploythreetypesofregularizationduringtraining:
Table2: TheTransformerachievesbetterBLEUscoresthanpreviousstate-of-the-artmodelsonthe
English-to-GermanandEnglish-to-Frenchnewstest2014testsatafractionofthetrainingcost.
BLEU TrainingCost(FLOPs)
Model

## En-De En-Fr En-De En-Fr

ByteNet[18] 23.75
Deep-Att+PosUnk[39] 39.2 1.0·1020
GNMT+RL[38] 24.6 39.92 2.3·1019 1.4·1020
ConvS2S[9] 25.16 40.46 9.6·1018 1.5·1020
MoE[32] 26.03 40.56 2.0·1019 1.2·1020
Deep-Att+PosUnkEnsemble[39] 40.4 8.0·1020
GNMT+RLEnsemble[38] 26.30 41.16 1.8·1020 1.1·1021
ConvS2SEnsemble[9] 26.36 41.29 7.7·1019 1.2·1021
Transformer(basemodel) 27.3 38.1 3.3·1018
Transformer(big) 28.4 41.8 2.3·1019
ResidualDropout Weapplydropout[33]totheoutputofeachsub-layer,beforeitisaddedtothe
sub-layerinputandnormalized. Inaddition,weapplydropouttothesumsoftheembeddingsandthe
positionalencodingsinboththeencoderanddecoderstacks. Forthebasemodel,weusearateof

## P =0.1.

drop
LabelSmoothing Duringtraining,weemployedlabelsmoothingofvalueϵ = 0.1[36]. This
hurtsperplexity,asthemodellearnstobemoreunsure,butimprovesaccuracyandBLEUscore.

## Results

6.1 MachineTranslation
OntheWMT2014English-to-Germantranslationtask,thebigtransformermodel(Transformer(big)
inTable2)outperformsthebestpreviouslyreportedmodels(includingensembles)bymorethan2.0
BLEU,establishinganewstate-of-the-artBLEUscoreof28.4. Theconfigurationofthismodelis
listedinthebottomlineofTable3. Trainingtook3.5dayson8P100GPUs. Evenourbasemodel
surpassesallpreviouslypublishedmodelsandensembles,atafractionofthetrainingcostofanyof
thecompetitivemodels.
OntheWMT2014English-to-Frenchtranslationtask,ourbigmodelachievesaBLEUscoreof41.0,
outperformingallofthepreviouslypublishedsinglemodels,atlessthan1/4thetrainingcostofthe
previousstate-of-the-artmodel. TheTransformer(big)modeltrainedforEnglish-to-Frenchused
dropoutrateP =0.1,insteadof0.3.
drop
Forthebasemodels,weusedasinglemodelobtainedbyaveragingthelast5checkpoints,which
werewrittenat10-minuteintervals. Forthebigmodels,weaveragedthelast20checkpoints. We
usedbeamsearchwithabeamsizeof4andlengthpenaltyα = 0.6[38]. Thesehyperparameters
werechosenafterexperimentationonthedevelopmentset. Wesetthemaximumoutputlengthduring
inferencetoinputlength+50,butterminateearlywhenpossible[38].
Table2summarizesourresultsandcomparesourtranslationqualityandtrainingcoststoothermodel
architecturesfromtheliterature. Weestimatethenumberoffloatingpointoperationsusedtotraina
modelbymultiplyingthetrainingtime,thenumberofGPUsused,andanestimateofthesustained
single-precisionfloating-pointcapacityofeachGPU5.
6.2 ModelVariations
ToevaluatetheimportanceofdifferentcomponentsoftheTransformer,wevariedourbasemodel
indifferentways,measuringthechangeinperformanceonEnglish-to-Germantranslationonthe
5Weusedvaluesof2.8,3.7,6.0and9.5TFLOPSforK80,K40,M40andP100,respectively.
Table3: VariationsontheTransformerarchitecture. Unlistedvaluesareidenticaltothoseofthebase
model. AllmetricsareontheEnglish-to-Germantranslationdevelopmentset,newstest2013. Listed
perplexitiesareper-wordpiece,accordingtoourbyte-pairencoding,andshouldnotbecomparedto
per-wordperplexities.
train PPL BLEU params
N d d h d d P ϵ
model ff k v drop ls steps (dev) (dev) ×106
base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65
1 512 512 5.29 24.9
4 128 128 5.00 25.5
(A)
16 32 32 4.91 25.8
32 16 16 5.01 25.4
16 5.16 25.1 58
(B)
32 5.01 25.4 60
2 6.11 23.7 36
4 5.19 25.3 50
8 4.88 25.5 80
(C) 256 32 32 5.75 24.5 28
1024 128 128 4.66 26.0 168
1024 5.12 25.4 53
4096 4.75 26.2 90
0.0 5.77 24.6
0.2 4.95 25.5
(D)
0.0 4.67 25.3
0.2 5.47 25.7
(E) positionalembeddinginsteadofsinusoids 4.92 25.7
big 6 1024 4096 16 0.3 300K 4.33 26.4 213
developmentset,newstest2013. Weusedbeamsearchasdescribedintheprevioussection,butno
checkpointaveraging. WepresenttheseresultsinTable3.
InTable3rows(A),wevarythenumberofattentionheadsandtheattentionkeyandvaluedimensions,
keeping the amount of computation constant, as described in Section 3.2.2. While single-head
attentionis0.9BLEUworsethanthebestsetting,qualityalsodropsoffwithtoomanyheads.
InTable3rows(B),weobservethatreducingtheattentionkeysized hurtsmodelquality. This
suggests that determining compatibility is not easy and that a more sophisticated compatibility
functionthandotproductmaybebeneficial. Wefurtherobserveinrows(C)and(D)that,asexpected,
biggermodelsarebetter,anddropoutisveryhelpfulinavoidingover-fitting.Inrow(E)wereplaceour
sinusoidalpositionalencodingwithlearnedpositionalembeddings[9],andobservenearlyidentical
resultstothebasemodel.
6.3 EnglishConstituencyParsing
ToevaluateiftheTransformercangeneralizetoothertasksweperformedexperimentsonEnglish
constituencyparsing. Thistaskpresentsspecificchallenges: theoutputissubjecttostrongstructural
constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence
modelshavenotbeenabletoattainstate-of-the-artresultsinsmall-dataregimes[37].
Wetraineda4-layertransformerwithd =1024ontheWallStreetJournal(WSJ)portionofthe
model
PennTreebank[25],about40Ktrainingsentences. Wealsotraineditinasemi-supervisedsetting,
usingthelargerhigh-confidenceandBerkleyParsercorporafromwithapproximately17Msentences
[37]. Weusedavocabularyof16KtokensfortheWSJonlysettingandavocabularyof32Ktokens
forthesemi-supervisedsetting.
Weperformedonlyasmallnumberofexperimentstoselectthedropout,bothattentionandresidual
(section5.4),learningratesandbeamsizeontheSection22developmentset,allotherparameters
remained unchanged from the English-to-German base translation model. During inference, we
Table4: TheTransformergeneralizeswelltoEnglishconstituencyparsing(ResultsareonSection23
ofWSJ)
Parser Training WSJ23F1
Vinyals&Kaiserelal. (2014)[37] WSJonly,discriminative 88.3
Petrovetal. (2006)[29] WSJonly,discriminative 90.4
Zhuetal. (2013)[40] WSJonly,discriminative 90.4
Dyeretal. (2016)[8] WSJonly,discriminative 91.7
Transformer(4layers) WSJonly,discriminative 91.3
Zhuetal. (2013)[40] semi-supervised 91.3
Huang&Harper(2009)[14] semi-supervised 91.3
McCloskyetal. (2006)[26] semi-supervised 92.1
Vinyals&Kaiserelal. (2014)[37] semi-supervised 92.1
Transformer(4layers) semi-supervised 92.7
Luongetal. (2015)[23] multi-task 93.0
Dyeretal. (2016)[8] generative 93.3
increasedthemaximumoutputlengthtoinputlength+300. Weusedabeamsizeof21andα=0.3
forbothWSJonlyandthesemi-supervisedsetting.
Our results in Table 4 show that despite the lack of task-specific tuning our model performs sur-
prisinglywell,yieldingbetterresultsthanallpreviouslyreportedmodelswiththeexceptionofthe
RecurrentNeuralNetworkGrammar[8].
IncontrasttoRNNsequence-to-sequencemodels[37],theTransformeroutperformstheBerkeley-
Parser[29]evenwhentrainingonlyontheWSJtrainingsetof40Ksentences.

## Conclusion

Inthiswork,wepresentedtheTransformer,thefirstsequencetransductionmodelbasedentirelyon
attention,replacingtherecurrentlayersmostcommonlyusedinencoder-decoderarchitectureswith
multi-headedself-attention.
For translation tasks, the Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014
English-to-Frenchtranslationtasks,weachieveanewstateoftheart. Intheformertaskourbest
modeloutperformsevenallpreviouslyreportedensembles.
Weareexcitedaboutthefutureofattention-basedmodelsandplantoapplythemtoothertasks. We
plantoextendtheTransformertoproblemsinvolvinginputandoutputmodalitiesotherthantextand
toinvestigatelocal,restrictedattentionmechanismstoefficientlyhandlelargeinputsandoutputs
suchasimages,audioandvideo. Makinggenerationlesssequentialisanotherresearchgoalsofours.
The code we used to train and evaluate our models is available at https://github.com/
tensorflow/tensor2tensor.
Acknowledgements WearegratefultoNalKalchbrennerandStephanGouwsfortheirfruitful
comments,correctionsandinspiration.

## References

[1] JimmyLeiBa,JamieRyanKiros,andGeoffreyEHinton. Layernormalization. arXivpreprint
arXiv:1607.06450,2016.
[2] DzmitryBahdanau,KyunghyunCho,andYoshuaBengio. Neuralmachinetranslationbyjointly
learningtoalignandtranslate. CoRR,abs/1409.0473,2014.
[3] DennyBritz,AnnaGoldie,Minh-ThangLuong,andQuocV.Le. Massiveexplorationofneural
machinetranslationarchitectures. CoRR,abs/1703.03906,2017.
[4] JianpengCheng,LiDong,andMirellaLapata. Longshort-termmemory-networksformachine
reading. arXivpreprintarXiv:1601.06733,2016.
[5] KyunghyunCho,BartvanMerrienboer,CaglarGulcehre,FethiBougares,HolgerSchwenk,
andYoshuaBengio. Learningphraserepresentationsusingrnnencoder-decoderforstatistical
machinetranslation. CoRR,abs/1406.1078,2014.
[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv
preprintarXiv:1610.02357,2016.
[7] JunyoungChung,ÇaglarGülçehre,KyunghyunCho,andYoshuaBengio. Empiricalevaluation
ofgatedrecurrentneuralnetworksonsequencemodeling. CoRR,abs/1412.3555,2014.
[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural
networkgrammars. InProc.ofNAACL,2016.
[9] JonasGehring,MichaelAuli,DavidGrangier,DenisYarats,andYannN.Dauphin. Convolu-
tionalsequencetosequencelearning. arXivpreprintarXiv:1705.03122v2,2017.
[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint
arXiv:1308.0850,2013.
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im-
age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition,pages770–778,2016.
[12] SeppHochreiter,YoshuaBengio,PaoloFrasconi,andJürgenSchmidhuber. Gradientflowin
recurrentnets: thedifficultyoflearninglong-termdependencies,2001.
[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,
9(8):1735–1780,1997.
[14] ZhongqiangHuangandMaryHarper. Self-trainingPCFGgrammarswithlatentannotations
acrosslanguages. InProceedingsofthe2009ConferenceonEmpiricalMethodsinNatural
LanguageProcessing,pages832–841.ACL,August2009.
[15] RafalJozefowicz,OriolVinyals,MikeSchuster,NoamShazeer,andYonghuiWu. Exploring
thelimitsoflanguagemodeling. arXivpreprintarXiv:1602.02410,2016.
[16] ŁukaszKaiserandSamyBengio. Canactivememoryreplaceattention? InAdvancesinNeural
InformationProcessingSystems,(NIPS),2016.
[17] ŁukaszKaiserandIlyaSutskever. NeuralGPUslearnalgorithms. InInternationalConference
onLearningRepresentations(ICLR),2016.
[18] NalKalchbrenner,LasseEspeholt,KarenSimonyan,AaronvandenOord,AlexGraves,andKo-
rayKavukcuoglu.Neuralmachinetranslationinlineartime.arXivpreprintarXiv:1610.10099v2,
2017.
[19] YoonKim,CarlDenton,LuongHoang,andAlexanderM.Rush. Structuredattentionnetworks.
InInternationalConferenceonLearningRepresentations,2017.
[20] DiederikKingmaandJimmyBa. Adam: Amethodforstochasticoptimization. InICLR,2015.
[21] OleksiiKuchaievandBorisGinsburg. FactorizationtricksforLSTMnetworks. arXivpreprint
arXiv:1703.10722,2017.
[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen
Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint
arXiv:1703.03130,2017.
[23] Minh-ThangLuong,QuocV.Le,IlyaSutskever,OriolVinyals,andLukaszKaiser. Multi-task
sequencetosequencelearning. arXivpreprintarXiv:1511.06114,2015.
[24] Minh-ThangLuong,HieuPham,andChristopherDManning. Effectiveapproachestoattention-
basedneuralmachinetranslation. arXivpreprintarXiv:1508.04025,2015.
[25] MitchellPMarcus,MaryAnnMarcinkiewicz,andBeatriceSantorini.Buildingalargeannotated
corpusofenglish: Thepenntreebank. Computationallinguistics,19(2):313–330,1993.
[26] DavidMcClosky,EugeneCharniak,andMarkJohnson. Effectiveself-trainingforparsing. In
ProceedingsoftheHumanLanguageTechnologyConferenceoftheNAACL,MainConference,
pages152–159.ACL,June2006.
[27] AnkurParikh,OscarTäckström,DipanjanDas,andJakobUszkoreit. Adecomposableattention
model. InEmpiricalMethodsinNaturalLanguageProcessing,2016.
[28] RomainPaulus,CaimingXiong,andRichardSocher. Adeepreinforcedmodelforabstractive
summarization. arXivpreprintarXiv:1705.04304,2017.
[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact,
and interpretable tree annotation. In Proceedings of the 21st International Conference on
ComputationalLinguisticsand44thAnnualMeetingoftheACL,pages433–440.ACL,July
2006.
[30] OfirPressandLiorWolf. Usingtheoutputembeddingtoimprovelanguagemodels. arXiv
preprintarXiv:1608.05859,2016.
[31] RicoSennrich,BarryHaddow,andAlexandraBirch. Neuralmachinetranslationofrarewords
withsubwordunits. arXivpreprintarXiv:1508.07909,2015.
[32] NoamShazeer,AzaliaMirhoseini,KrzysztofMaziarz,AndyDavis,QuocLe,GeoffreyHinton,
andJeffDean. Outrageouslylargeneuralnetworks: Thesparsely-gatedmixture-of-experts
layer. arXivpreprintarXiv:1701.06538,2017.
[33] NitishSrivastava,GeoffreyEHinton,AlexKrizhevsky,IlyaSutskever,andRuslanSalakhutdi-
nov. Dropout: asimplewaytopreventneuralnetworksfromoverfitting. JournalofMachine
LearningResearch,15(1):1929–1958,2014.
[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory
networks. InC.Cortes, N.D.Lawrence, D.D.Lee, M.Sugiyama, andR.Garnett, editors,
AdvancesinNeuralInformationProcessingSystems28,pages2440–2448.CurranAssociates,
Inc.,2015.
[35] IlyaSutskever,OriolVinyals,andQuocVVLe. Sequencetosequencelearningwithneural
networks. InAdvancesinNeuralInformationProcessingSystems,pages3104–3112,2014.
[36] ChristianSzegedy,VincentVanhoucke,SergeyIoffe,JonathonShlens,andZbigniewWojna.
Rethinkingtheinceptionarchitectureforcomputervision. CoRR,abs/1512.00567,2015.
[37] Vinyals&Kaiser, Koo, Petrov, Sutskever, andHinton. Grammarasaforeignlanguage. In
AdvancesinNeuralInformationProcessingSystems,2015.
[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
Macherey,MaximKrikun,YuanCao,QinGao,KlausMacherey,etal. Google’sneuralmachine
translationsystem: Bridgingthegapbetweenhumanandmachinetranslation. arXivpreprint
arXiv:1609.08144,2016.
[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with
fast-forwardconnectionsforneuralmachinetranslation. CoRR,abs/1606.04199,2016.
[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate
shift-reduceconstituentparsing. InProceedingsofthe51stAnnualMeetingoftheACL(Volume
1: LongPapers),pages434–443.ACL,August2013.
Input-Input Layer5
AttentionVisualizations
Figure 3: An example of the attention mechanism following long-distance dependencies in the
encoderself-attentioninlayer5of6. Manyoftheattentionheadsattendtoadistantdependencyof
theverb‘making’,completingthephrase‘making...moredifficult’. Attentionshereshownonlyfor
theword‘making’. Differentcolorsrepresentdifferentheads. Bestviewedincolor.
siht
siht
tirips
tirips
taht
taht
ytirojam
ytirojam
naciremA
naciremA
stnemnrevog
stnemnrevog
evah
evah
dessap
dessap
wen
wen
swal
swal
ecnis
ecnis
9002
9002
gnikam
gnikam
eht
eht
noitartsiger
noitartsiger
gnitov
gnitov
ssecorp
ssecorp
erom
erom
tluciffid
tluciffid

## >Soe<



## >Soe<

>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
>dap<
Input-Input Layer5
Input-Input Layer5
Figure4: Twoattentionheads,alsoinlayer5of6,apparentlyinvolvedinanaphoraresolution. Top:
Fullattentionsforhead5. Bottom: Isolatedattentionsfromjusttheword‘its’forattentionheads5
and6. Notethattheattentionsareverysharpforthisword.
ehT
ehT
ehT
ehT
waL
waL
waL
waL
lliw
lliw
lliw
lliw
reven
reven
reven
reven
tcefrep
tcefrep
tcefrep
tcefrep
tub
tub
tub
tub
sti
sti
sti
sti
noitacilppa
noitacilppa
noitacilppa
noitacilppa
dluohs
dluohs
dluohs
dluohs
tsuj
tsuj
tsuj
tsuj
siht
siht
siht
siht
tahw
tahw
tahw
tahw
era
era
era
era
gnissim
gnissim
gnissim
gnissim
noinipo
noinipo
noinipo
noinipo

## >Soe<



## >Soe<



## >Soe<



## >Soe<

>dap<
>dap<
>dap<
>dap<
Input-Input Layer5
Input-Input Layer5
Figure5: Manyoftheattentionheadsexhibitbehaviourthatseemsrelatedtothestructureofthe
sentence. Wegivetwosuchexamplesabove,fromtwodifferentheadsfromtheencoderself-attention
atlayer5of6. Theheadsclearlylearnedtoperformdifferenttasks.
ehT
ehT
ehT
ehT
waL
waL
waL
waL
lliw
lliw
lliw
lliw
reven
reven
reven
reven
tcefrep
tcefrep
tcefrep
tcefrep
tub
tub
tub
tub
sti
sti
sti
sti
noitacilppa
noitacilppa
noitacilppa
noitacilppa
dluohs
dluohs
dluohs
dluohs
tsuj
tsuj
tsuj
tsuj
siht
siht
siht
siht
tahw
tahw
tahw
tahw
era
era
era
era
gnissim
gnissim
gnissim
gnissim
noinipo
noinipo
noinipo
noinipo

## >Soe<



## >Soe<



## >Soe<



## >Soe<

>dap<
>dap<
>dap<
>dap<

## References

1. [1] JimmyLeiBa,JamieRyanKiros,andGeoffreyEHinton. Layernormalization. arXivpreprint arXiv:1607.06450,2016.
2. [2] DzmitryBahdanau,KyunghyunCho,andYoshuaBengio. Neuralmachinetranslationbyjointly learningtoalignandtranslate. CoRR,abs/1409.0473,2014.
3. [3] DennyBritz,AnnaGoldie,Minh-ThangLuong,andQuocV.Le. Massiveexplorationofneural machinetranslationarchitectures. CoRR,abs/1703.03906,2017.
4. [4] JianpengCheng,LiDong,andMirellaLapata. Longshort-termmemory-networksformachine reading. arXivpreprintarXiv:1601.06733,2016. 10
5. [5] KyunghyunCho,BartvanMerrienboer,CaglarGulcehre,FethiBougares,HolgerSchwenk, andYoshuaBengio. Learningphraserepresentationsusingrnnencoder-decoderforstatistical machinetranslation. CoRR,abs/1406.1078,2014.
6. [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprintarXiv:1610.02357,2016.
7. [7] JunyoungChung,ÇaglarGülçehre,KyunghyunCho,andYoshuaBengio. Empiricalevaluation ofgatedrecurrentneuralnetworksonsequencemodeling. CoRR,abs/1412.3555,2014.
8. [8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural networkgrammars. InProc.ofNAACL,2016.
9. [9] JonasGehring,MichaelAuli,DavidGrangier,DenisYarats,andYannN.Dauphin. Convolu- tionalsequencetosequencelearning. arXivpreprintarXiv:1705.03122v2,2017.
10. [10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850,2013.
11. [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im- age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,pages770–778,2016.
12. [12] SeppHochreiter,YoshuaBengio,PaoloFrasconi,andJürgenSchmidhuber. Gradientflowin recurrentnets: thedifficultyoflearninglong-termdependencies,2001.
13. [13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780,1997.
14. [14] ZhongqiangHuangandMaryHarper. Self-trainingPCFGgrammarswithlatentannotations acrosslanguages. InProceedingsofthe2009ConferenceonEmpiricalMethodsinNatural LanguageProcessing,pages832–841.ACL,August2009.
15. [15] RafalJozefowicz,OriolVinyals,MikeSchuster,NoamShazeer,andYonghuiWu. Exploring thelimitsoflanguagemodeling. arXivpreprintarXiv:1602.02410,2016.
16. [16] ŁukaszKaiserandSamyBengio. Canactivememoryreplaceattention? InAdvancesinNeural InformationProcessingSystems,(NIPS),2016.
17. [17] ŁukaszKaiserandIlyaSutskever. NeuralGPUslearnalgorithms. InInternationalConference onLearningRepresentations(ICLR),2016.
18. [18] NalKalchbrenner,LasseEspeholt,KarenSimonyan,AaronvandenOord,AlexGraves,andKo- rayKavukcuoglu.Neuralmachinetranslationinlineartime.arXivpreprintarXiv:1610.10099v2,
19. 2017.
20. [19] YoonKim,CarlDenton,LuongHoang,andAlexanderM.Rush. Structuredattentionnetworks. InInternationalConferenceonLearningRepresentations,2017.
21. [20] DiederikKingmaandJimmyBa. Adam: Amethodforstochasticoptimization. InICLR,2015.
22. [21] OleksiiKuchaievandBorisGinsburg. FactorizationtricksforLSTMnetworks. arXivpreprint arXiv:1703.10722,2017.
23. [22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130,2017.
24. [23] Minh-ThangLuong,QuocV.Le,IlyaSutskever,OriolVinyals,andLukaszKaiser. Multi-task sequencetosequencelearning. arXivpreprintarXiv:1511.06114,2015.
25. [24] Minh-ThangLuong,HieuPham,andChristopherDManning. Effectiveapproachestoattention- basedneuralmachinetranslation. arXivpreprintarXiv:1508.04025,2015. 11
26. [25] MitchellPMarcus,MaryAnnMarcinkiewicz,andBeatriceSantorini.Buildingalargeannotated corpusofenglish: Thepenntreebank. Computationallinguistics,19(2):313–330,1993.
27. [26] DavidMcClosky,EugeneCharniak,andMarkJohnson. Effectiveself-trainingforparsing. In ProceedingsoftheHumanLanguageTechnologyConferenceoftheNAACL,MainConference, pages152–159.ACL,June2006.
28. [27] AnkurParikh,OscarTäckström,DipanjanDas,andJakobUszkoreit. Adecomposableattention model. InEmpiricalMethodsinNaturalLanguageProcessing,2016.
29. [28] RomainPaulus,CaimingXiong,andRichardSocher. Adeepreinforcedmodelforabstractive summarization. arXivpreprintarXiv:1705.04304,2017.
30. [29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on ComputationalLinguisticsand44thAnnualMeetingoftheACL,pages433–440.ACL,July
31. 2006.
32. [30] OfirPressandLiorWolf. Usingtheoutputembeddingtoimprovelanguagemodels. arXiv preprintarXiv:1608.05859,2016.
33. [31] RicoSennrich,BarryHaddow,andAlexandraBirch. Neuralmachinetranslationofrarewords withsubwordunits. arXivpreprintarXiv:1508.07909,2015.
34. [32] NoamShazeer,AzaliaMirhoseini,KrzysztofMaziarz,AndyDavis,QuocLe,GeoffreyHinton, andJeffDean. Outrageouslylargeneuralnetworks: Thesparsely-gatedmixture-of-experts layer. arXivpreprintarXiv:1701.06538,2017.
35. [33] NitishSrivastava,GeoffreyEHinton,AlexKrizhevsky,IlyaSutskever,andRuslanSalakhutdi- nov. Dropout: asimplewaytopreventneuralnetworksfromoverfitting. JournalofMachine LearningResearch,15(1):1929–1958,2014.
36. [34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. InC.Cortes, N.D.Lawrence, D.D.Lee, M.Sugiyama, andR.Garnett, editors, AdvancesinNeuralInformationProcessingSystems28,pages2440–2448.CurranAssociates, Inc.,2015.
37. [35] IlyaSutskever,OriolVinyals,andQuocVVLe. Sequencetosequencelearningwithneural networks. InAdvancesinNeuralInformationProcessingSystems,pages3104–3112,2014.
38. [36] ChristianSzegedy,VincentVanhoucke,SergeyIoffe,JonathonShlens,andZbigniewWojna. Rethinkingtheinceptionarchitectureforcomputervision. CoRR,abs/1512.00567,2015.
39. [37] Vinyals&Kaiser, Koo, Petrov, Sutskever, andHinton. Grammarasaforeignlanguage. In AdvancesinNeuralInformationProcessingSystems,2015.
40. [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey,MaximKrikun,YuanCao,QinGao,KlausMacherey,etal. Google’sneuralmachine translationsystem: Bridgingthegapbetweenhumanandmachinetranslation. arXivpreprint arXiv:1609.08144,2016.
41. [39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forwardconnectionsforneuralmachinetranslation. CoRR,abs/1606.04199,2016.
42. [40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduceconstituentparsing. InProceedingsofthe51stAnnualMeetingoftheACL(Volume 1: LongPapers),pages434–443.ACL,August2013. 12 Input-Input Layer5 AttentionVisualizations Figure 3: An example of the attention mechanism following long-distance dependencies in the encoderself-attentioninlayer5of6. Manyoftheattentionheadsattendtoadistantdependencyof theverb‘making’,completingthephrase‘making...moredifficult’. Attentionshereshownonlyfor theword‘making’. Differentcolorsrepresentdifferentheads. Bestviewedincolor. 13 tI tI si si ni ni siht siht tirips tirips taht taht a a ytirojam ytirojam fo fo naciremA naciremA stnemnrevog stnemnrevog evah evah dessap dessap wen wen swal swal ecnis ecnis 9002 9002 gnikam gnikam eht eht noitartsiger noitartsiger ro ro gnitov gnitov ssecorp ssecorp erom erom tluciffid tluciffid . . >SOE< >SOE< >dap< >dap< >dap< >dap< >dap< >dap< >dap< >dap< >dap< >dap< >dap< >dap< Input-Input Layer5 Input-Input Layer5 Figure4: Twoattentionheads,alsoinlayer5of6,apparentlyinvolvedinanaphoraresolution. Top: Fullattentionsforhead5. Bottom: Isolatedattentionsfromjusttheword‘its’forattentionheads5 and6. Notethattheattentionsareverysharpforthisword. 14 ehT ehT ehT ehT waL waL waL waL lliw lliw lliw lliw reven reven reven reven eb eb eb eb tcefrep tcefrep tcefrep tcefrep , , , , tub tub tub tub sti sti sti sti noitacilppa noitacilppa noitacilppa noitacilppa dluohs dluohs dluohs dluohs eb eb eb eb tsuj tsuj tsuj tsuj - - - - siht siht siht siht si si si si tahw tahw tahw tahw ew ew ew ew era era era era gnissim gnissim gnissim gnissim , , , , ni ni ni ni ym ym ym ym noinipo noinipo noinipo noinipo . . . . >SOE< >SOE< >SOE< >SOE< >dap< >dap< >dap< >dap< Input-Input Layer5 Input-Input Layer5 Figure5: Manyoftheattentionheadsexhibitbehaviourthatseemsrelatedtothestructureofthe sentence. Wegivetwosuchexamplesabove,fromtwodifferentheadsfromtheencoderself-attention atlayer5of6. Theheadsclearlylearnedtoperformdifferenttasks. 15 ehT ehT ehT ehT waL waL waL waL lliw lliw lliw lliw reven reven reven reven eb eb eb eb tcefrep tcefrep tcefrep tcefrep , , , , tub tub tub tub sti sti sti sti noitacilppa noitacilppa noitacilppa noitacilppa dluohs dluohs dluohs dluohs eb eb eb eb tsuj tsuj tsuj tsuj - - - - siht siht siht siht si si si si tahw tahw tahw tahw ew ew ew ew era era era era gnissim gnissim gnissim gnissim , , , , ni ni ni ni ym ym ym ym noinipo noinipo noinipo noinipo . . . . >SOE< >SOE< >SOE< >SOE< >dap< >dap< >dap< >dap<

---
*Processed on 2025-08-07 18:19:42 UTC*
*Processing time: 5.03 seconds*