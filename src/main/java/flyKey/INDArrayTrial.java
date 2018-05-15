package flyKey;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class INDArrayTrial {
	public static char space = ' ';
	public static String strg1 = "Der ";
	public static String strg2 = "Cottbuser ";
	public static String strg3 = "Postkutscher ";
	public static String strg4 = "putzt ";
	public static String strg5 = "den ";
	public static String strg6 = "Cottbuser ";
	public static String strg7 = "Postkutschkasten ";
	
	public static char[][] arrayOfChars = {strg1.toCharArray(), strg2.toCharArray(), strg3.toCharArray(), strg4.toCharArray(), 
			strg5.toCharArray(), strg6.toCharArray(), strg7.toCharArray()};
	
	// define a sentence to learn.
    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
	//private static final char[] LEARNSTRING ;

	// a list of all possible characters
	//private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<>();
	
	private static final List<Character> defaultCharacterList = new ArrayList<>();
	private static final String BACKUP_MODEL_FILENAME = "/home/samuel/eclipseWorkspace/flyKey/rnn_train_MLN.bak.zip"; // filename of the previous version of the model (backup)
    private static final String MODEL_FILENAME = "/home/samuel/eclipseWorkspace/flyKey/rnn_train_MLN.zip"; // filename of the model
	// RNN dimensions
	private static final int HIDDEN_LAYER_WIDTH = 50;
	private static final int HIDDEN_LAYER_CONT = 2;
	//private static final Random r = new Random(7894);
	private MultiLayerNetwork net;
	char[] LEARNSTRING;
	
	public void main(String[] args) {
		
		//System.out.println("DefaultCharSet " + defaultCharacterList.size() + " LEARNSTRING.length " + LEARNSTRING.length);

				File networkFile = new File(toTempPath(MODEL_FILENAME));
		        //int offset = 0;
		        if (networkFile.exists()) {
		            System.out.println("Loading the existing network...");
		            
		            try {
						net = ModelSerializer.restoreMultiLayerNetwork(networkFile);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
		            
		            System.out.print("Enter a word to start predicting or 0 to train a new word: ");
		            String input;
		            
		            try (Scanner scanner = new Scanner(System.in)) {
		                
		            	input = scanner.nextLine();
		                if (input.toLowerCase().equals("0")) {
		                	
		                    //startDialog(scanner);
		                	input = scanner.nextLine();
		                	trainModel(networkFile, input.toCharArray());
		                	
		                } else {
		                    //offset = Integer.valueOf(input);
		                    //test();
		                	
		                	autocompleteWord(input.toCharArray());
		                }
		            }
		            
		        } else {
		            System.out.println("Creating a new network...");
		            
		            try {
						networkFile.createNewFile();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
		            
		            trainModel(networkFile, null);
		        }
				/*
				 * CREATE OUR TRAINING DATA
				 */
				// create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
				// SEQUENCE_POSITION

		
		//INDArray input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		
	}
	
	public void trainModel(File networkFile, char[] STRING) {
		
		//arrayOfChars = {strg1.toCharArray(), strg2.toCharArray()};
		
				// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<>();
				//for (char c : LEARNSTRING)
				//			LEARNSTRING_CHARS.add(c);
				
				//LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);
				
				//System.out.println("LEARNSTRING_CHARS_LIST " + LEARNSTRING_CHARS_LIST);
				
		for(char c : getDefaultCharacterSet()) {
			LEARNSTRING_CHARS.add(c);
					//defaultCharacterList.add(c);
		}
				
		defaultCharacterList.addAll(LEARNSTRING_CHARS);
				
		System.out.println("DefaultCharSet " + defaultCharacterList);
		
		
		int loopCycles;
		
		if(STRING == null) {
			loopCycles = arrayOfChars.length;
			
			// some common parameters
			NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
			builder.seed(123);
			builder.biasInit(0);
			builder.miniBatch(false);
			builder.updater(new RmsProp(0.001));
			builder.weightInit(WeightInit.XAVIER);

			ListBuilder listBuilder = builder.list();

			// first difference, for rnns we need to use GravesLSTM.Builder
			for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
				GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
				hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
				hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
				// adopted activation function from GravesLSTMCharModellingExample
				// seems to work well with RNNs
				hiddenLayerBuilder.activation(Activation.TANH);
				listBuilder.layer(i, hiddenLayerBuilder.build());
			}

			// we need to use RnnOutputLayer for our RNN
			RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
			// softmax normalizes the output neurons, the sum of all outputs is 1
			// this is required for our sampleFromDistribution-function
			outputLayerBuilder.activation(Activation.SOFTMAX);
			outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
			outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
			listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

			// finish builder
			listBuilder.pretrain(false);
			listBuilder.backprop(true);

			// create network
			MultiLayerConfiguration conf = listBuilder.build();
			net = new MultiLayerNetwork(conf);
			net.init();
			net.setListeners(new ScoreIterationListener(1));
			
		} else {
			loopCycles = 1;
		}
		
		int i = 0;
		
		while(i < loopCycles) {
			
			if(STRING == null) {
				LEARNSTRING = arrayOfChars[i];
			} else {
				LEARNSTRING = STRING;
			}
			
			System.out.println("begining the training " + LEARNSTRING.toString());
				
			INDArray input = Nd4j.zeros(1, defaultCharacterList.size(), LEARNSTRING.length);
			INDArray labels = Nd4j.zeros(1, defaultCharacterList.size(), LEARNSTRING.length);
			
			// loop through our sample-sentence
			int samplePos = 0;
					
			for (char currentChar : LEARNSTRING) {
				// small hack: when currentChar is the last, take the first char as
				// nextChar - not really required. Added to this hack by adding a starter first character.
				char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
						
				// input neuron for current-char is 1 at "samplePos"
				input.putScalar(new int[] { 0, defaultCharacterList.indexOf(currentChar), samplePos }, 1);
						
				// output neuron for next-char is 1 at "samplePos"
						
				//labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
				labels.putScalar(new int[] { 0, defaultCharacterList.indexOf(nextChar), samplePos }, 1);
				samplePos++;
			}
			
			DataSet trainingData = new DataSet(input, labels);
	
			// some epochs
			for (int epoch = 0; epoch < 1000; epoch++) {
	
				System.out.println("Epoch " + epoch);
	
				// train the data
				net.fit(trainingData);
	
				// clear current stance from the last example
				net.rnnClearPreviousState();
	
			}
			
			i++;
			
			try {
				
				ModelSerializer.writeModel(net, networkFile, true);
				saveModel(networkFile);
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 
		}
		
		//main(null);
		
	}
	
	public void autocompleteWord(char[] word4Autocomplete){
		
		LEARNSTRING = word4Autocomplete;
		
		// put the first character into the rrn as an initialisation
		//INDArray testInit = Nd4j.zeros(defaultCharacterList.size());
		INDArray testInit = Nd4j.zeros(LEARNSTRING.length, defaultCharacterList.size());
		
		for(int i = 0; i < LEARNSTRING.length; i++) {
			//int[] inte1 = new int[]{0,defaultCharacterList.indexOf(LEARNSTRING[0])};
			int[] inte1 = new int[]{i,defaultCharacterList.indexOf(LEARNSTRING[i])};
			//int[] inte2 = new int[]{1,defaultCharacterList.indexOf(LEARNSTRING[1])};
			
			testInit.putScalar(inte1, 1);
			//testInit.putScalar(inte2, 1);
			//testInit.putScalar(defaultCharacterList.indexOf(LEARNSTRING[1]), 1);
		}
		
		//int testInitIdx = Nd4j.getExecutioner().exec(new IMax(testInit), 1).getInt(0);
		//int testInitSec = Nd4j.getExecutioner().exec(new IMax(testInit), 1).getInt(1);
		
		// print the chosen output
        //System.out.println(defaultCharacterList.get(testInitIdx) + defaultCharacterList.get(testInitSec));
        //System.out.println(defaultCharacterList.get(testInitIdx).toString() + 
        //		defaultCharacterList.get(testInitSec).toString());
        		
		// run one step -> IMPORTANT: rnnTimeStep() must be called, not
		// output()
		// the output shows what the net thinks what should come next
		INDArray output = net.rnnTimeStep(testInit);
		
		//output.length()
		//System.out.println(output.rows());
		int sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);
		
		System.out.print(defaultCharacterList.get(sampledCharacterIdx).toString());
		// now the net should guess LEARNSTRING.length more characters
        for (int j = 0; defaultCharacterList.get(sampledCharacterIdx) != 
        		defaultCharacterList.get(defaultCharacterList.indexOf(space)) ; j++) {
        	
        	if(j == 0) {
        		net.rnnClearPreviousState();
        	}
            // use the last output as input
            INDArray nextInput = Nd4j.zeros(defaultCharacterList.size());
            //INDArray nextInput = Nd4j.zeros(2, defaultCharacterList.size());
            
            nextInput.putScalar(sampledCharacterIdx, 1);
            
            //int[] next1 = new int[]{0,sampledCharacterIdx};
			//int[] next2 = new int[]{1,sampledCharacterSec};
			
			//testInit.putScalar(next1, 1);
			//testInit.putScalar(next2, 1);
            
            output = net.rnnTimeStep(nextInput);
        	
            // first process the last output of the network to a concrete
            // neuron, the neuron with the highest output has the highest
            // chance to get chosen
            sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);
            //int sampledCharacterSec = Nd4j.getExecutioner().exec(new IMax(output), 0).getInt(0);

            // print the chosen output
            System.out.print(defaultCharacterList.get(sampledCharacterIdx).toString()); 
            //		defaultCharacterList.get(sampledCharacterSec).toString());

        }
		System.out.print("\n");
		//main(null);
	}
	
	/** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
	public static char[] getMinimalCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c='a'; c<='z'; c++) validChars.add(c);
		for(char c='A'; c<='Z'; c++) validChars.add(c);
		for(char c='0'; c<='9'; c++) validChars.add(c);
		char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
		for( char c : temp ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}

	/** As per getMinimalCharacterSet(), but with a few extra characters */
	public static char[] getDefaultCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c : getMinimalCharacterSet() ) validChars.add(c);
		char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
				'\\', '|', '<', '>'};
		for( char c : additionalChars ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}
	
	private void saveModel(File networkFile) throws IOException {
        System.out.println("Saving the model...");
        File backup = new File(toTempPath(BACKUP_MODEL_FILENAME));
        if (networkFile.exists()) {
            if (backup.exists()) {
                backup.delete();
            }
            networkFile.renameTo(backup);
        }
        ModelSerializer.writeModel(net, networkFile, true);
        System.out.println("Done.");
    }
	
	private String toTempPath(String path) {
    	//logger.log(Level.CONFIG, System.getProperty("java.io.tmpdir"));
        //return System.getProperty("java.io.tmpdir") + "/" + path;
    	return path;
    }
}