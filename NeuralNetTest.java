package Neurons;
import Neurons.*;
import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.junit.Ignore;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetTest {
	/***Test Data*****/	
	private int numInput = 2;
	private int numHidden = 4;
	private int numOutput = 1;
	private double u_inputData[][] = {{1,1},{1,0},{0,1},{0,0}};	
	private double u_expectedOutput[][] = {{0},{1},{1},{0}};
	private double learningRate = 0.2;
	private double momentumRate_1 = 0.0;
	private double u_lowerBound = 0.0;
	private double u_upperBound = 1.0;
	private double b_inputData[][] = {{-1,-1},{1,-1},{-1,1},{1,1}};	
	private double b_expectedOutput[][] = {{-1},{1},{1},{-1}};
	private double momentumRate_2 = 0.9;
	private double b_lowerBound = -1.0;
	private double b_upperBound = 1.0;
	
	private ArrayList<Double> errorInEachEpoch;
	
	Neuron testNeuron = new Neuron("test");
	@Before
	public void setUp() throws Exception {

	}
	/*
	 * This test tests the binary Sigmoid function
	 */
	@Ignore("Ignored")
	@Test
	public void testBinarySigmoid() {
		double x = 1;
		double expectedResult = 0.731058;
		double delta = 0.001;
		double actualResult = testNeuron.unipolarSigmoid(x);
		assertEquals(expectedResult, actualResult, delta);	
	}
	/*
	 * This test tests the bipolar Sigmoid function
	 */
	@Ignore("Ignored")
	@Test
	public void testBipolarSigmoid() {
		double x = 1;
		double expectedResult = 0.462117;
		double delta = 0.001;
		double actualResult = testNeuron.bipolarSigmoid(x);
		assertEquals(expectedResult, actualResult, delta);
	}
	/***
	 * This test tests the derivative function of bipolar sigmoid
	 */
	@Ignore("Ignored")
	@Test
	public void testBipolarSigmoidDerivative() {
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_1,b_lowerBound,b_upperBound,1);

		double x = 0.5;
		double expectedResult = 0.375;
		double delta = 0.001;
		double actualResult = testNeuronNet.bipolarSigmoidDerivative(x);
		assertEquals(expectedResult, actualResult, delta);
		
	}
	
	/**
	 * This test tests the derivative of flexible sigmoid function 
	 */
	@Ignore("Ignored")
	@Test
	public void testCustomizedSigmoidDerivative() {
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_1,u_lowerBound,u_upperBound,1);
		double x = 0.5;
		double expectedResult = 0.25;
		double delta = 0.001;
		double actualResult = testNeuronNet.customSigmoidDerivative(x);
		assertEquals(expectedResult, actualResult, delta);		
	}
	
	/***
	 * This test tests the random weight generator 
	 */
	@Ignore("Ignored")
	@Test
	public void testRandomWeightGenerator() {
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_1,u_lowerBound,u_upperBound,1);
		double actualResult = testNeuronNet.getRandom(-0.5, 0.5);
		System.out.println(actualResult);
		assertTrue("The output is out of range:"+actualResult, -0.5<=actualResult&&actualResult<=0.5);		
	}
	
	/*
	 * This test tests the converge of neural network using binary representation.
	 */
	@Ignore("Ignored")
	@Test
	public void testUnipolarConverge() {
		//testNeuronNet.zeroWeights();
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_1,u_lowerBound,u_upperBound,1);
		try {
		this.tryConverge(testNeuronNet,u_inputData,u_expectedOutput,10000, 0.05);
		testNeuronNet.printRunResults(this.errorInEachEpoch, "unipolar.csv");
		}
		catch(IOException e){
			System.out.println(e);
		}
		
	}
	
	/*
	 * This test tests the converge of neural network using bipolar representation.
	 */
	@Ignore("Ignored")
	@Test
	public void testBipolarConverge() {
		//testNeuronNet.zeroWeights();
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_1,b_lowerBound,b_upperBound,1);
		try {
		this.tryConverge(testNeuronNet,b_inputData,b_expectedOutput,10000, 0.05);
		testNeuronNet.printRunResults(this.errorInEachEpoch, "bipolar.csv");
		}
		catch(IOException e){
			System.out.println(e);
		}		
	}
	/*
	 * This test tests the converge of neural network using bipolar representation with momentum.
	 */
	@Ignore("Ignored")
	@Test
	public void testBipolarWithMomentumConverge() {
		//testNeuronNet.zeroWeights();
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_2,b_lowerBound,b_upperBound,1);
		try {
		this.tryConverge(testNeuronNet,b_inputData,b_expectedOutput,10000, 0.05);
		testNeuronNet.printRunResults(this.errorInEachEpoch, "bipolarMomentum.csv");
		File file = new File("Weight_"+testNeuronNet.getNetID()+".txt");
		file.createNewFile();
		testNeuronNet.save(file);
		}
		catch(IOException e){
			System.out.println(e);
		}		
	}
	
	/**
	 * This test tests the average converge performance of bipolar neural network.
	 */

	@Ignore("Ignored")
	@Test
	public void testUnipolarAverage(){
		int average = EpochAverage(momentumRate_1,u_lowerBound,u_upperBound,u_inputData,u_expectedOutput,0.05,10000,500);
		System.out.println("The average of number of epoches to converge is: "+average+"\n");
	}
	


	/**
	 * This test tests the average converge performance of unipolar neural network.
	 */
	//@Ignore("Ignored")
	@Test
	public void testBipolarAverage(){
		int average = EpochAverage(momentumRate_1,b_lowerBound,b_upperBound,b_inputData,b_expectedOutput,0.05,10000,1000);
		System.out.println("The average of number of epoches to converge is: "+average+"\n");
	}
	
	/**
	 * This test tests the average converge performance of bipolar neural network with momentum acceleration.
	 */
	@Ignore("Ignored")
	@Test
	public void testBipolarWithMomentumAverage(){
		int average = EpochAverage(momentumRate_2,b_lowerBound,b_upperBound,b_inputData,b_expectedOutput,0.05,10000,1000);
		System.out.println("The average of number of epoches to converge is: "+average+"\n");
	}
	@Ignore("Ignored")
	@Test
	public void testLoadingWeight(){
		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate_2,b_lowerBound,b_upperBound,1);
		try {
			//File file = new File("E:\\Work\\java\\RoboCode_RL_NN\\weight.txt");	
			File file = new File("Weight_1.txt");
			testNeuronNet.load(file);			
		}
		catch(IOException e){
			System.out.println(e);
		}		
	}
	/***
	 * This function calculates the average of amount of epoch that one trial of network training takes, 
	 * it take in parameters of a neural network and returns the average of epoch number
	 * @param momentum
	 * @param lowerbound
	 * @param upperbound
	 * @param input
	 * @param expected
	 * @param minError
	 * @param maxSteps
	 * @param numTrials
	 * @return the average of number of epochs
	 */
	public int EpochAverage(double momentum, double lowerbound, double upperbound, double[][] input, double[][] expected,double minError, int maxSteps, int numTrials) {
		int epochNumber, failure,success;
		double average = 0f;
		epochNumber = 0;
		failure = 0;
		success = 0;
		NeuralNet testNeuronNet = null;
		for(int i = 0; i < numTrials; i++) {
			testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentum,lowerbound,upperbound,1); //Construct a new neural net object
			tryConverge(testNeuronNet,input,expected,maxSteps, minError);//Train the network with step and error constrains
			epochNumber = getErrorArray().size(); //get the epoch number of this trial.
			if( epochNumber < maxSteps) {
				average = average +  epochNumber;
				success ++; 
			}
			else {
				failure++;
			}			
		}
		double convergeRate = 100*success/(success+failure);
		System.out.println("The net converges for "+convergeRate+" percent of the time.\n" );
		average = average/success;		
		return (int)average;		
	}	
	/**
	 * This method run train for many epochs till the NN converge subjects to the max step constrain.	
	 * @param maxStep
	 * @param minError
	 */
	public void tryConverge(NeuralNet theNet, double[][] input, double [][] expected,int maxStep, double minError) {
		int i;
		double totalerror = 1;
		errorInEachEpoch = new ArrayList<>();
		for(i = 0; i < maxStep && totalerror > minError; i++) {
			totalerror = 0.0;
			for(int j = 0; j < input.length; j++) {
				totalerror += theNet.train(input[j],expected[j]);				
			}
			totalerror = totalerror*0.5;
			errorInEachEpoch.add(totalerror);
		}
		System.out.println("Sum of squared error in last epoch = " + totalerror);
		System.out.println("Number of epoch: "+ i + "\n");
		if(i == maxStep) {
			System.out.println("Error in training, try again!");
		}		
	}
	
	public ArrayList <Double> getErrorArray(){
		return errorInEachEpoch;
	}
	
	public void setErrorArray(ArrayList<Double> errors) {
		errorInEachEpoch = errors;
	}

}
