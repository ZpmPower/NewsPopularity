using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.IO;
using CNTK;
namespace NewsPopularity
{
    public partial class TrainNewsDataCategory
    {
        static string strNewsFilePath = "NewsCNTKModel";
        public static void SaveTrainedModel(Function model, string fileName)
        {
            model.Save(fileName);
        }
        public static void TrainNewsCategoryByMinibatchSource(DeviceDescriptor device, bool resetPreviousStateIfExist = true)
        {

            var dataFolder = "Data";//files must be on the same folder as program
            var dataPath = Path.Combine(dataFolder, "trainNews_cntk.txt");


            var featureStreamName = "features";
            var labelsStreamName = "label";

            //Network definition
            int inputDim = 21; //входные параметры
            int numOutputClasses = 6; //выходные параметры - категории
            int numHiddenLayers = 1; //количество спрятаных слоев
            int hidenLayerDim = 21; //нейронов в скрытом слое
            uint sampleSize = 30;

            var streamConfig = new StreamConfiguration[]
               {
                   new StreamConfiguration(featureStreamName, inputDim),
                   new StreamConfiguration(labelsStreamName, numOutputClasses)
               };
            // build a NN model
            //define input and output variable and connecting to the stream configuration
            var feature = Variable.InputVariable(new NDShape(1, inputDim), DataType.Float, featureStreamName);
            var label = Variable.InputVariable(new NDShape(1, numOutputClasses), DataType.Float, labelsStreamName);

            //Build simple Feed Froward Neural Network model
            // var ffnn_model = CreateMLPClassifier(device, numOutputClasses, hidenLayerDim, feature, classifierName);
            var activation = Activation.Tanh;
            var ffnn_model = createFFNN(feature, numHiddenLayers, hidenLayerDim, numOutputClasses, activation, "CategoryNNModel", device);

            //Loss and error functions definition
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(ffnn_model), label, "lossFunction");
            var classError = CNTKLib.ClassificationError(new Variable(ffnn_model), label, "classificationError");

            // set learning rate for the network
            double learnParameter = 0.0011125;
            var learningRatePerSample = new TrainingParameterScheduleDouble(learnParameter, 1);

            //define learners for the NN model

            var learner = Learner.SGDLearner(ffnn_model.Parameters(), learningRatePerSample);

            //define trainer based on ffnn_model, loss and error functions , and SGD learner 
            var trainer = Trainer.CreateTrainer(ffnn_model, trainingLoss, classError, new Learner[] { learner });

            //restoring the state of the trainer, in order to continue training instead of training from the beginning 
            if (!resetPreviousStateIfExist)
            {
                if (File.Exists(strNewsFilePath))
                    trainer.RestoreFromCheckpoint(strNewsFilePath);
            }

            // prepare the training data
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                dataPath, streamConfig, MinibatchSource.InfinitelyRepeat, true);
            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);



            //Preparation for the iterative learning process
            //used 800 epochs/iterations. Batch size will be the same as sample size since the data set is small
            int epochs = 800;
            int epochs1 = epochs;
            int i = 0;
            while (epochs > -1)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(sampleSize, device);
                //pass to the trainer the current batch separated by the features and label.
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { feature, minibatchData[featureStreamInfo] },
                    { label, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);


                TestHelper.PrintTrainingProgress(trainer, i++, 500);

                // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                if (minibatchData.Values.Any(a => a.sweepEnd))
                {
                    epochs--;
                }
            }
            //Summary of training
            double acc = Math.Round((1.0 - trainer.PreviousMinibatchEvaluationAverage()) * 100, 2);
            Console.WriteLine($"------TRAINING SUMMARY--------");
            Console.WriteLine($"The model trained with the accuracy {acc}%");
            var tuple = Tuple.Create(numHiddenLayers, hidenLayerDim, learnParameter, epochs1, sampleSize, activation);

            //save model in ckp format for later training
            Console.WriteLine("Would you like to persist the model so that it can be trained again! (y-yes, otherwise No)");
            if (Console.ReadLine() == "y")
                trainer.SaveCheckpoint(strNewsFilePath);

            //// validate the model
            Console.WriteLine("Would you like to evaluate the model (y-yes, otherwise No)!");
            if (Console.ReadLine() == "y")
            {
                var modelPath = Path.Combine(dataFolder, "modelFileName");
                ffnn_model.Save(modelPath);
                EvaluateIrisModel(modelPath, device);
            }
        }
        private static void EvaluateIrisModelMyData(string modelFileName, DeviceDescriptor device)
        {
            //calculate Iris flow from those dimensions
            //Example: 5.0f, 3.5f, 1.3f, 0.3f, setosa
            float[] xVal = new float[21] { 0.568421051f, 0.999999995f, 0.759090906f, 0.033338999f,
                0.547323842f, 0.033336982f, 0.352664128f, 0.03333605f, 0.465529756f, 0.154168306f, 0.052356021f, 0.020942408f,
                0.714285714f, 0.285714286f, 0.348782468f, 0.0625f, 1f, 0.5f, 0.6f, 0f, 0.6f };

            //|features 0.568421051 0.999999995 0.759090906 0.033338999 0.547323842 0.033336982 0.352664128 0.03333605 0.465529756 0.154168306 0.052356021 0.020942408 0.714285714 0.285714286 0.348782468 0.0625 1 0.5 0.6 0 0.6 |label 0 1 0 0 0 0
            //load the model from disk
            var ffnn_model = Function.Load(modelFileName, device);

            //extract features and label from the model
            Variable feature = ffnn_model.Arguments[0];
            Variable label = ffnn_model.Output;

            Value xValues = Value.CreateBatch<float>(new int[] { feature.Shape[0] }, xVal, device);
            //Value yValues = - we don't need it, because we are going to calculate it

            //map the variables and values
            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(feature, xValues);
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(label, null);

            //evaluate the model
            ffnn_model.Evaluate(inputDataMap, outputDataMap, device);
            //extract the result  as one hot vector
            var outputData = outputDataMap[label].GetDenseData<float>(label);

            //transforms into class value
            var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();
            var flower = actualLabels.FirstOrDefault();
            string inputs = String.Empty;
            foreach (var x in xVal)
            {
                inputs += x + ",";
            }
            var strFlower = flower == 0 ? "setosa" : flower == 1 ? "versicolor" : "versicolor";
            Console.WriteLine($"Model Prediction: Input({inputs}), Iris Flower={flower}");
            Console.WriteLine($"Model Expectation: Input({inputs}), Category= 0 1 0 0 0 0");

            return;

        }
        private static void EvaluateIrisModel(string modelFileName, DeviceDescriptor device)
        {
            var dataFolder = "Data";//files must be on the same folder as program
            var trainPath = Path.Combine(dataFolder, "testNews_cntk.txt");
            var featureStreamName = "features";
            var labelsStreamName = "label";
            var ffnn_model = Function.Load(modelFileName, device);

            //extract features and label from the model
            var feature = ffnn_model.Arguments[0];
            var label = ffnn_model.Output;
            
            //stream configuration to distinct features and labels in the file
            var streamConfig = new StreamConfiguration[]
               {
                   new StreamConfiguration(featureStreamName, feature.Shape[0]),
                   new StreamConfiguration(labelsStreamName, label.Shape[0])
               };

            // prepare testing data
            var testMinibatchSource = MinibatchSource.TextFormatMinibatchSource(
                trainPath, streamConfig, MinibatchSource.InfinitelyRepeat, true);
            var featureStreamInfo = testMinibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = testMinibatchSource.StreamInfo(labelsStreamName);

            int batchSize = 5;
            int miscountTotal = 0, totalCount = 5;
            while (true)
            {
                var minibatchData = testMinibatchSource.GetNextMinibatch((uint)batchSize, device);
                if (minibatchData == null || minibatchData.Count == 0)
                    break;
                totalCount += (int)minibatchData[featureStreamInfo].numberOfSamples;

                // expected labels are in the mini batch data.
                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(label);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();
                var inputDataMap = new Dictionary<Variable, Value>() {
                    { feature, minibatchData[featureStreamInfo].data }
                };

                var outputDataMap = new Dictionary<Variable, Value>() {
                    { label, null }
                };

                ffnn_model.Evaluate(inputDataMap, outputDataMap, device);
                var outputData = outputDataMap[label].GetDenseData<float>(label);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                miscountTotal += misMatches;
               
                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Mis-classify Count = {miscountTotal}");

                if (totalCount >= 5000)
                    break;
            }
            Console.WriteLine($"---------------");
            Console.WriteLine($"------TESTING SUMMARY--------");
            float accuracy = (1.0F - (float)miscountTotal / (float)totalCount);
            Console.WriteLine($"Model Accuracy = {accuracy}");
            var resultPath = Path.Combine(dataFolder, "resultNews_cntk.txt");
            var valuesPath = Path.Combine(dataFolder, "valuesResNews_cntk.txt");
            using (StreamWriter sw = new StreamWriter(resultPath, true, System.Text.Encoding.Default))
            {
                //sw.WriteLine("numHiddenLayers = {0} hidenLayerDim = {1} learnParameter = {2} epochs = {3} sampleSize = {4} activation = {5} accuracy = {6} trainAcc = {7}", 
                //    tuple.Item1, tuple.Item2, tuple.Item3, tuple.Item4, tuple.Item5, tuple.Item6.ToString(), accuracy, acc);     
            }
            return;
        }
        public enum Activation
        {
            None,
            ReLU,
            Sigmoid,
            Tanh
        }
        static Function ApplyActivationFunction(Function layer, Activation actFun)
        {
            switch (actFun)
            {
                default:
                case Activation.None:
                    return layer;
                case Activation.ReLU:
                    return CNTKLib.ReLU(layer);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(layer);
                case Activation.Tanh:
                    return CNTKLib.Tanh(layer);
            }
        }
        static Function simpleLayer(Function input, int outputDim, DeviceDescriptor device)
        {
            //prepare default parameters values
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1);

            //create weight and bias vectors
            var var = (Variable)input;
            var shape = new int[] { outputDim, var.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "w");
            //var biasParam = new Parameter(new NDShape(1, outputDim), 0, device, "b");

            //construct W * X + b matrix
            return CNTKLib.Times(weightParam, input); //+ biasParam;
        }
        static Function createFFNN(Variable input, int hiddenLayerCount, int hiddenDim, int outputDim, Activation activation, string modelName, DeviceDescriptor device)
        {
            //First the parameters initialization must be performed
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1);

            //hidden layers creation
            //first hidden layer
            Function h = simpleLayer(input, hiddenDim, device);
            h = ApplyActivationFunction(h, activation);
            //2,3, ... hidden layers
            for (int i = 1; i < hiddenLayerCount; i++)
            {
                h = simpleLayer(h, hiddenDim, device);
                h = ApplyActivationFunction(h, activation);
            }
            //the last action is creation of the output layer
            var r = simpleLayer(h, outputDim, device);
            r.SetName(modelName);
            return r;
        }
        public static Function LoadTrainedModel(string fileName, DeviceDescriptor device)
        {
            return Function.Load(fileName, device, ModelFormat.CNTKv2);
        }
        static void Main(string[] args)
        {
            var cpu = DeviceDescriptor.UseDefaultDevice();
            TrainNewsCategoryByMinibatchSource(cpu);
            EvaluateIrisModelMyData("Data/modelFileName",cpu);
        }
    }
}
