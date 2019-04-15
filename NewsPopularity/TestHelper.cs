using CNTK;
using ExcelDna.Integration;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NewsPopularity
{
    
    public class TestHelper
    {
        
        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }

        public static float ValidateModelWithMinibatchSource(
            string modelFile, MinibatchSource testMinibatchSource,
            int[] imageDim, int numClasses, string featureInputName, string labelInputName, string outputName,
            DeviceDescriptor device, int maxCount = 1000)
        {
            Function model = Function.Load(modelFile, device);
            var imageInput = model.Arguments[0];
            var labelOutput = model.Outputs.Single(o => o.Name == outputName);

            var featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName);
            var labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName);

            int batchSize = 50;
            int miscountTotal = 0, totalCount = 50;
            while (true)
            {
                var minibatchData = testMinibatchSource.GetNextMinibatch((uint)batchSize, device);
                if (minibatchData == null || minibatchData.Count == 0)
                    break;
                totalCount += (int)minibatchData[featureStreamInfo].numberOfSamples;

                // expected labels are in the minibatch data.
                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(labelOutput);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                var inputDataMap = new Dictionary<Variable, Value>() {
                    { imageInput, minibatchData[featureStreamInfo].data }
                };

                var outputDataMap = new Dictionary<Variable, Value>() {
                    { labelOutput, null }
                };

                model.Evaluate(inputDataMap, outputDataMap, device);
                var outputData = outputDataMap[labelOutput].GetDenseData<float>(labelOutput);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                miscountTotal += misMatches;
                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");

                if (totalCount > maxCount)
                    break;
            }

            float errorRate = 1.0F * miscountTotal / totalCount;
            Console.WriteLine($"Model Validation Error = {errorRate}");
            return errorRate;
        }

        public static bool MiniBatchDataIsSweepEnd(ICollection<MinibatchData> minibatchValues)
        {
            return minibatchValues.Any(a => a.sweepEnd);
        }

        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }
        public static void Normalize(string input, string output)
        {
            List<Tuple<string,double>> maxValues = new List<Tuple<string,double>>();
            List<string> normList = new List<string>();
            StreamReader sr = new StreamReader("Data/maxValues.txt");
            string line;
            while (!sr.EndOfStream)
            {
                line = sr.ReadLine();
                string[] strs = line.Split(' ');
                float value = float.Parse(strs[1], CultureInfo.InvariantCulture.NumberFormat);
                Tuple<string, double> tuple = new Tuple<string, double>(strs[0], value);
                maxValues.Add(tuple);
            }
            StreamReader ss = new StreamReader(input);
            while (!ss.EndOfStream)
            {
                line = ss.ReadLine();
                string substr = "|features ";
                string substr1 = "|label";
                int n = line.IndexOf(substr);
                int n1 = line.IndexOf(substr1);
                line = line.Remove(n, substr.Length);
                string label = line.Substring(n1 - 11);
                line = line.Remove(n1 - 11);

                string[] arStr = line.Split(' ');
                List<string> s = arStr.Cast<string>().ToList();
                string vector = "|features ";
                int i = 0;
                foreach (string str in s)
                {
                    double newValue = float.Parse(str, CultureInfo.InvariantCulture.NumberFormat) / maxValues[i].Item2;
                    i++;
                    vector += String.Format("{0:0.000000000000}", ((double)newValue)) + ' ';
                }
                vector += label;
                normList.Add(vector);
            }
            try
            {
                using (StreamWriter sw = new StreamWriter(output, true, System.Text.Encoding.Default))
                {
                    foreach(string s in normList)
                    sw.WriteLine(s);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
        public static void PrintOutputDims(Function function, string functionName)
        {
            NDShape shape = function.Output.Shape;

            if (shape.Rank == 3)
            {
                Console.WriteLine($"{functionName} dim0: {shape[0]}, dim1: {shape[1]}, dim2: {shape[2]}");
            }
            else
            {
                Console.WriteLine($"{functionName} dim0: {shape[0]}");
            }
        }
        public class IrisModel
        {
            [ExcelFunction(Description = "NewsEval - Prediction for the News Category based on 26 input values.")]
            public static string IrisEval(object arg)
            {
                try
                {
                    //First convert object in to array
                    object[,] obj = (object[,])arg;

                    //create list to convert values
                    List<float> calculatedOutput = new List<float>();
                    //
                    foreach (var s in obj)
                    {
                        var ss = float.Parse(s.ToString(), CultureInfo.InvariantCulture);
                        calculatedOutput.Add(ss);
                    }
                    if (calculatedOutput.Count != 26)
                        throw new Exception("Incorrect number of input variables. It must be 26!");
                    return EvaluateModel(calculatedOutput.ToArray());
                }
                catch (Exception ex)
                {
                    return ex.Message;
                }

            }
            private static string EvaluateModel(float[] vector)
            {
                //load the model from disk
                var ffnn_model = Function.Load("Data/modelFileName", DeviceDescriptor.CPUDevice);

                //extract features and label from the model
                Variable feature = ffnn_model.Arguments[0];
                Variable label = ffnn_model.Output;

                Value xValues = Value.CreateBatch<float>(new int[] { feature.Shape[0] }, vector, DeviceDescriptor.CPUDevice);
                //Value yValues = - we don't need it, because we are going to calculate it

                //map the variables and values
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(feature, xValues);
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(label, null);

                //evaluate the model
                ffnn_model.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
                //extract the result  as one hot vector
                var outputData = outputDataMap[label].GetDenseData<float>(label);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();
                var category = actualLabels.FirstOrDefault();
                return category.ToString();
            }
        }
    }
}