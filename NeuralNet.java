package Neurons;

import java.io.*;

import java.util.ArrayList;
import java.util.List;

import java.util.Random;

import interfaces.NeuralNetInterface;

public class NeuralNet implements NeuralNetInterface {
	
	private int netID;
	private int argNumInputs;
	private int argNumHiddens;
	private int argNumOutputs;
	private double argLearningRate;
	private double argMomentumRate;
	private double argQMin;
	private double argQMax;
	/*Keep inputNeuron, hiddenNeuron, outputNeuron separately in arraylists*/
	private ArrayList<Neuron> inputLayerNeurons = new ArrayList<Neuron>();	
	private ArrayList<Neuron> hiddenLayerNeurons = new ArrayList<Neuron>();
	private ArrayList<Neuron> outputLayerNeurons = new ArrayList<Neuron>();
	
	/* Need to keep in mind that, all neurons can connect to the same bias neuron, since the output of bias neuron is not evaluated*/
	private Neuron biasNeuron = new Neuron("bias"); // Neuron id 0 is reserved for bias neuron
	/**These arrays and list save the input data, expected output, actual output and error in each epoch****/
	//private double inputData[][];	
	//private double expectedOutput[][];
	//private double epochOutput[][];//Initial value -1 for each output
	//private ArrayList<Double> errorInEachEpoch = new ArrayList<>();
	
	
	public NeuralNet(
					int numInputs, int numHiddens, 
					int numOutputs, double learningRate, 
					double momentumRate, double a, double b,
					int id
					) {
		this.argNumInputs = numInputs;
		this.argNumHiddens = numHiddens;
		this.argNumOutputs = numOutputs;
		this.argLearningRate = learningRate;
		this.argMomentumRate = momentumRate;
		this.argQMin = a;
		this.argQMax = b;
		//this.inputData = inputData;
		//this.expectedOutput = expectedOutput;
		this.setUpNetwork();
		this.initializeWeights();
		this.netID = id;
	}
	
	public void setUpNetwork() {
		/*Set up Input layer first*/
		for(int i = 0; i < this.argNumInputs;i++) {
			String index = "Input"+Integer.toString(i);
			//System.out.println(index);
			Neuron neuron = new Neuron(index);
			inputLayerNeurons.add(neuron);
		}
		
		/*Set up hidden layer*/
		for(int j = 0; j < this.argNumHiddens;j++) {
			String index = "Hidden"+Integer.toString(j);
			//System.out.println(index);
			Neuron neuron = new Neuron(index,"Customized",inputLayerNeurons,biasNeuron);
			hiddenLayerNeurons.add(neuron);
		}
		/*Set up output layer*/
		for(int k = 0; k < this.argNumOutputs;k++) {
			String index = "Output"+Integer.toString(k);
			//System.out.println(index);
			Neuron neuron = new Neuron(index,"Customized",hiddenLayerNeurons,biasNeuron);
			outputLayerNeurons.add(neuron);
		}
	}
	/**
	 * This method sets input value in each forwarding.
	 * @param X The input vector. An array of doubles.
	 */
	public void setInputData(double [] inputs) {
		for(int i = 0; i < inputLayerNeurons.size(); i++) {
			inputLayerNeurons.get(i).setOutput(inputs[i]);//Input Layer Neurons only have output values.
		}
		biasNeuron.setOutput(1.0);
	}
	
	/**
	 * Get the output values from all output neurons, only one output in our problem
	 * @return a double output[]
	 */
	public double[] getOutputResults() {
		double [] outputs = new double[outputLayerNeurons.size()];
		for(int i = 0; i < outputLayerNeurons.size(); i++) {
			outputs[i] = outputLayerNeurons.get(i).getOutput();
		}
		return outputs;
	}
	
	public int getNetID(){
		return this.netID;
	}
	
	/*****
	 * This method calculates the output of the NN based on the input 
	 * vector using forward propagation, calculation is done layer by layer 
	 */
	public void forwardPropagation() {
		for(Neuron hidden: hiddenLayerNeurons) {
			hidden.calculateOutput(argQMin,argQMax);
		}
		
		for (Neuron output: outputLayerNeurons) {
			output.calculateOutput(argQMin, argQMax);
		}
	}
	
	public ArrayList<Neuron> getInputNeurons(){
		return this.inputLayerNeurons;
	}
	
	public ArrayList<Neuron> getHiddenNeurons(){
		return this.hiddenLayerNeurons;
	}
	
	public ArrayList<Neuron> getOutputNeurons(){
		return this.outputLayerNeurons;
	}
	
	/**
	 * 
	 * @return an array of results for each forwarding in a single epoch
	 */
	/*public double [][] getEpochResults() {
		return epochOutput;
	}
	
	public void setEpochResults(double[][] results){
		for(int i = 0; i < results.length;i++) {
			for(int j = 0; j < results[i].length;j++)
			{
				epochOutput[i][j] = results[i][j];
			}
		}
	}*/
	
	/**
	 * This perform backpropagation to update all the weight in this NN.
	 * @param expectedOutput
	 */
	private void applyBackpropagation(double expectedOutput[]) {
		int i = 0;
		for(Neuron output : outputLayerNeurons) {
			double yi = output.getOutput();
			double ci = expectedOutput[i];			
			ArrayList<NeuronConnection> connections = output.getInputConnectionList();
			double error = customSigmoidDerivative(yi)*(ci-yi);//Calculate delta for this neuron and record the error for later use
			output.setError(error);
			for(NeuronConnection link : connections) {
				double xi = link.getInput();
				double deltaWeight = argLearningRate*error*xi + argMomentumRate*link.getDeltaWeight();//Calculate the delta weight for the specific connection
				double newWeight = link.getWeight() + deltaWeight;//Calculate new weight
				link.setDeltaWeight(deltaWeight);//Update weight and delta weight 
				link.setWeight(newWeight);			
			}//Weights for all input connection of this output neuron is updated. 
			i++;
		}//Update weights for all output neurons, in this problem there is only one output
		
		for(Neuron hidden: hiddenLayerNeurons) {
			ArrayList<NeuronConnection> connections = hidden.getInputConnectionList();
			double yi =hidden.getOutput();
			double sumWeightedError= 0.0;
			for(Neuron output: outputLayerNeurons) {
				double wjh = output.getInputConnection(hidden.getId()).getWeight();//Get the weight of the output connection that connects to this neuron
				double errorFromAbove = output.getError();//Get the delta of the output neuron that connects to this neuron
				sumWeightedError = sumWeightedError + wjh *errorFromAbove;
			}//Sum of weighted error is calculated
			double error = customSigmoidDerivative(yi)*sumWeightedError;//Calculate and record the delta for this hidden neuron
			hidden.setError(error);
			for(NeuronConnection link : connections) {
				double xi = link.getInput();				
				double deltaWeight = argLearningRate*error*xi + argMomentumRate * link.getDeltaWeight();
				double newWeight = link.getWeight() + deltaWeight;
				link.setDeltaWeight(deltaWeight);
				link.setWeight(newWeight);							
			}//Calculate delta weight for each input connection to this neuron and update the weights.
		}		
	}
	/***
	 * This method set input data to the NN, run one forward propagation and return the output for this run. 
	 */
	@Override
	public double [] outputFor(double[] inputData) {
		setInputData(inputData);
		forwardPropagation();
		double outputs[] = getOutputResults();
		return outputs;
	}

	@Override
	/**
	 * This method performs one epoch of train to the NN.
	 * @return accumulate squared error generated in one epoch.
	 */
	public double train(double [] argInputVector, double [] argTargetOutput) {
			double error = 0.0;
			double output[] = outputFor(argInputVector);
			for (int j = 0; j < argTargetOutput.length; j++) {
				double deltaErr = Math.pow((output[j]-argTargetOutput[j]),2);
				error = error + deltaErr;//sum of error for all  output neurons
			}		
			this.applyBackpropagation(argTargetOutput);
		//errorInEachEpoch.add(0.5*totalError);
		return error;
	}
	
	
	/**
	 * This method print epoch errors into a .csv file.
	 * @param errors
	 * @param fileName
	 * @throws IOException
	 */
	public void printRunResults(ArrayList<Double> errors, String fileName) throws IOException {
		int epoch;
		PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));
		printWriter.printf("Epoch Number, Total Squared Error, \n");
		for(epoch = 0; epoch < errors.size(); epoch++) {
			printWriter.printf("%d, %f, \n", epoch, errors.get(epoch));
		}
		printWriter.flush();
		printWriter.close();
	}
	

	@Override
	public void save(File argFile) {
		PrintStream savefile = null;
		try{
			savefile = new PrintStream(new FileOutputStream(argFile) );
			savefile.println(outputLayerNeurons.size());
			savefile.println(hiddenLayerNeurons.size());
			savefile.println(inputLayerNeurons.size());
			for(Neuron output : outputLayerNeurons){
				ArrayList<NeuronConnection> connections = output.getInputConnectionList();
				for(NeuronConnection link : connections){
					savefile.println(link.getWeight());
				}
			}
			for(Neuron hidden: hiddenLayerNeurons) {
				ArrayList<NeuronConnection> connections = hidden.getInputConnectionList();
				for(NeuronConnection link : connections){
					savefile.println(link.getWeight());
				}
			}
			savefile.flush();
			savefile.close();				
		}
		catch(IOException e){
			System.out.println("Cannot save the weight table.");
		}

	}

	@Override
	public void load(File argFileName) throws IOException {
		
		try{
			BufferedReader readfile = new BufferedReader(new FileReader(argFileName));
			int numOutputNeuron = Integer.valueOf(readfile.readLine());
			int numHiddenNeuron = Integer.valueOf(readfile.readLine());
			int numInputNeuron = Integer.valueOf(readfile.readLine());
			if ( numInputNeuron != inputLayerNeurons.size() ) {
				System.out.println ( "*** Number of inputs in file does not match expectation");
				readfile.close();
				throw new IOException();
			}
			if ( numHiddenNeuron != hiddenLayerNeurons.size() ) {
				System.out.println ( "*** Number of hidden in file does not match expectation" );
				readfile.close();
				throw new IOException();
			}
			if ( numOutputNeuron != outputLayerNeurons.size() ) {
				System.out.println ( "*** Number of output in file does not match expectation" );
				readfile.close();
				throw new IOException();
			}			

			for(Neuron output : outputLayerNeurons){
				ArrayList<NeuronConnection> connections = output.getInputConnectionList();
				for(NeuronConnection link : connections){
					link.setWeight(Double.valueOf(readfile.readLine()));
				}
			}
			for(Neuron hidden: hiddenLayerNeurons) {
				ArrayList<NeuronConnection> connections = hidden.getInputConnectionList();
				for(NeuronConnection link : connections){
					link.setWeight(Double.valueOf(readfile.readLine()));
				}
			}
			
			readfile.close();
		}
		catch(IOException e){
			System.out.println("IOException failed to open reader: " + e);
		}
		
	}
	
	public double sigmoidDerivative(double yi) {
		double result = yi*(1 - yi);
		return result;
	}
	
	public double bipolarSigmoidDerivative(double yi) {
		double result = 1.0/2.0 * (1-yi) * (1+yi);
		return result;
	}
	 /**
     * This method implements the first derivative of the customized sigmoid
     * @param x The input
     * @return f'(x) = -(1 / (b - a))(customSigmoid - a)(customSigmoid - b)
     */	
	public double customSigmoidDerivative(double yi) {
		double result = -(1.0/(argQMax-argQMin)) * (yi-argQMin) * (yi-argQMax);
		return result;
	}

	@Override
	public void initializeWeights() {
		// TODO Auto-generated method stub
		double upperbound = 0.5;
		double lowerbound = -0.5;
		for(Neuron neuron: hiddenLayerNeurons) {
			ArrayList <NeuronConnection> connections = neuron.getInputConnectionList();
			for(NeuronConnection connect: connections) {
				connect.setWeight(getRandom(lowerbound,upperbound));
			}
		}
		for(Neuron neuron: outputLayerNeurons) {
			ArrayList <NeuronConnection> connections = neuron.getInputConnectionList();
			for(NeuronConnection connect: connections) {
				connect.setWeight(getRandom(lowerbound,upperbound));
			}
		}
	}

	@Override
	public void zeroWeights() {
		for(Neuron neuron: hiddenLayerNeurons) {
			ArrayList <NeuronConnection> connections = neuron.getInputConnectionList();
			for(NeuronConnection connect: connections) {
				connect.setWeight(0);
			}
		}
		for(Neuron neuron:outputLayerNeurons) {
			ArrayList <NeuronConnection> connections = neuron.getInputConnectionList();
			for(NeuronConnection connect: connections) {
				connect.setWeight(0);
			}
		}

	}
	
	public double getRandom(double lowerbound, double upperbound) {
		double random = Math.random();	
		double result = lowerbound+(upperbound-lowerbound)*random;
		return result;
	}
	
	

}
