package flyKey;

import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.examples.recurrent.character.GravesLSTMCharModellingExample;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

/**
 * This example trains a RNN. When trained we only have to put the first
 * character of LEARNSTRING to the RNN, and it will recite the following chars
 *
 * @author Peter Grossmann
 */
public class CompGraphRNN {

	// define a sentence to learn.
    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
	//private static final char[] LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray();
	private static final char[] LEARNSTRING = "Postkutscher".toCharArray();

	// a list of all possible characters
	private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<>();

	// RNN dimensions
	private static final int HIDDEN_LAYER_WIDTH = 50;
	private static final int HIDDEN_LAYER_CONT = 2;
    private static final Random r = new Random(7894);
    static CharIterator iter;

	public static void main(String[] args) {
		
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
        //int miniBatchSize = 32;						//Size of mini batch to use when  training
        //int exampleLength = 9;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        //int numEpochs = 5;							//Total number of training epochs
        //int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        //int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        //int nCharactersToSample = 300;				//Length of each sample to generate
        //String generationInitialization = "alguie";		//Optional character initialization; a random character is used if null
        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        //CharacterIterator iter = GravesLSTMCharModellingExample.getShakespeareIterator(miniBatchSize, exampleLength);
        
        int miniBatchSize = LEARNSTRING.length;
		
        try {
			iter = new CharIterator(LEARNSTRING, Charset.forName("UTF-8"), miniBatchSize, 
					1, validCharacters, new Random(12345));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            //.seed(12345)
        	.miniBatch(false)
        	//.seed(123)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp(0.1))
            .graphBuilder()
            .addInputs("input") //Give the input a name. For a ComputationGraph with multiple inputs, this also defines the input array orders
            //First layer: name "first", with inputs from the input called "input"
            .addLayer("first", new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .activation(Activation.TANH).build(),"input")
            //Second layer, name "second", with inputs from the layer called "first"
            .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build(),"first")
            //Output layer, name "outputlayer" with inputs from the two layers called "first" and "second"
            .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(2*lstmLayerSize).nOut(nOut).build(),"first","second")
            .setOutputs("outputLayer")  //List the output. For a ComputationGraph with multiple outputs, this also defines the input array orders
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        int totalNumParams = 0;
        for( int i=0; i<net.getNumLayers(); i++ ){
            int nParams = net.getLayer(i).numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

		// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		//------------------------------------------------------------------------------------------------
		
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<>();
		for (char c : LEARNSTRING)
			LEARNSTRING_CHARS.add(c);
		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

		// some common parameters
		
		//NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		//builder.seed(123);
		//builder.biasInit(0);
		//builder.miniBatch(false);
		//builder.updater(new RmsProp(0.001));
		//builder.weightInit(WeightInit.XAVIER);

		//ListBuilder listBuilder = builder.list();

		// first difference, for rnns we need to use GravesLSTM.Builder
		
		//for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
		//	GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
		//	hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
		//	hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			// adopted activation function from GravesLSTMCharModellingExample
			// seems to work well with RNNs
		//	hiddenLayerBuilder.activation(Activation.TANH);
		//	listBuilder.layer(i, hiddenLayerBuilder.build());
		//}

		// we need to use RnnOutputLayer for our RNN
		
		//RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		
		// softmax normalizes the output neurons, the sum of all outputs is 1
		// this is required for our sampleFromDistribution-function
		
		//outputLayerBuilder.activation(Activation.SOFTMAX);
		//outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		//outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
		//listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		// finish builder
		
		//listBuilder.pretrain(false);
		//listBuilder.backprop(true);

		// create network
		
		//MultiLayerConfiguration conf = listBuilder.build();
		//MultiLayerNetwork net = new MultiLayerNetwork(conf);
		//net.init();
		//net.setListeners(new ScoreIterationListener(1));

		/*
		 * CREATE OUR TRAINING DATA
		 */
		// create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
		// SEQUENCE_POSITION
		
		//INDArray input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		//INDArray labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		
		// loop through our sample-sentence
		
		//int samplePos = 0;
		//for (char currentChar : LEARNSTRING) {
			// small hack: when currentChar is the last, take the first char as
			// nextChar - not really required. Added to this hack by adding a starter first character.
		//	char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
			// input neuron for current-char is 1 at "samplePos"
		//	input.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			// output neuron for next-char is 1 at "samplePos"
		//	labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
		//	samplePos++;
		//}
		//--------------------------------------------------------------------------------------------------------------
		
		//DataSet trainingData = new DataSet(input, labels);

		// some epochs
		for (int epoch = 0; epoch < 200; epoch++) {

			System.out.println("Epoch " + epoch);

			// train the data
			net.fit(iter.next());

			// clear current stance from the last example
			net.rnnClearPreviousState();

			// put the first character into the rrn as an initialisation
			
			INDArray testInit = Nd4j.create(new int[]{miniBatchSize, validCharacters.length, 1}, 'f');
			
			//INDArray testInit = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			//testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);
			
			testInit.putScalar(new int[]{0, CharIterator.convertCharacterToIndex(LEARNSTRING[0]),1}, 1.0);
			
			// run one step -> IMPORTANT: rnnTimeStep() must be called, not
			// output()
			// the output shows what the net thinks what should come next
			INDArray output = net.rnnTimeStep(testInit)[0];
			output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output
			
			// now the net should guess LEARNSTRING.length more characters
            for (char dummy : LEARNSTRING) {

                // first process the last output of the network to a concrete
                // neuron, the neuron with the highest output has the highest
                // chance to get chosen
                int sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);

                // print the chosen output
                //System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));
                System.out.print(validCharacters[sampledCharacterIdx]);

                // use the last output as input
                //INDArray nextInput = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
                
                INDArray nextInput = Nd4j.create(new int[]{miniBatchSize, validCharacters.length, 1}, 'f');
                
                //nextInput.putScalar(sampledCharacterIdx, 1);
                
                nextInput.putScalar(new int[]{0, sampledCharacterIdx,1}, 1.0);
                
                output = net.rnnTimeStep(nextInput)[0];

            }
			System.out.print("\n");
		}
	}
}
