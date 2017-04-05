# G-eRules streaming Algorithm
G-eRules is a streaming classification algorithm. the G stands for the the use of the Gaussian distribution and the algorithm incorporate a new method for continuous attribute. Data stream classification is one of the most important DSM technique allowing to classifiy previously unseen data isntances. Different to traditinal classifiers for static data, data stream classifiers need to adapt to concept changes (concept drift) in the stream in real-time in order to reflect the most recent concept in the data as accurately as possible.

The implementation was used to carry out the experiment as presented in:
> _"Thien Le, Frederic Stahl, Joao Gomes, Mohamed Medhat Gaber, Giuseppe Di Fatta (2014) Computationally Efficient Rule-Based Classification for Continuous Streaming Data. In Proceedings of the Thirty-Fourth SGAI International Conference on Innovative Techniques and Applications of Artificial Intelligence, Cambridge. Springer, pp 21-34. ISBN 978-3-319-12068-3 doi: 10.1007/978-3-319-12069-0_2"._


## Getting Started
The algorithm was developed to integrate into MOA (Massive Online Analysis) for classification task. However, it is possible run the algorithm as a stand-alone application but you sill need to supple a data instance in form of a Instance class from Weka/ MOA.

If this project is included on classpath when openning MOA then we should be able to see the classifer from the list of classfier and can excute an evaluation like other classifiers in MOA. Any issues whe compiling the algorithm to use in MOA GUI can be raised here and we will try to address it as soon as we can.

### Running and Evauate the Algorithm as a Library
This section shows few examples how you can use the algorithm in your own project and examime the learned rules or make a prediction for an unseen data instance.

The algorithm make use of some data structures from WEKA project such as Instance, Instances, Attribute, etc... More information about these data structue can be found here:

[https://weka.wikispaces.com](https://weka.wikispaces.com)

You can also find guides about create ARFF file on-the-fly or covert csv format to ARFF, and utimately to Instances and Instance object from the WEKA's wikipage above.

#### Dependencies

* MOA (11.2014)
* Weka-package (version that is associated with MOA (11.2014))
* Commons Math 3.6

#### Create a Rules Library from ARFF
The following example shows how to load data from an ARFF and create a rules library. 

```java
// read arff file WEKA way
public static void main(String[] args) throws Exception {

    // read arff file WEKA way
    DataSource source = new DataSource("data/cmc.arff");

    // HoeffdingRules classifier
    GeRules gErules = new GeRules();
    gErules.prepareForUse();

    // load data into instances set
    Instances data = source.getDataSet();

    // setting class attribute if the data format does not provide this information
    // For example, the XRFF format saves the class attribute information as well
    if (data.classIndex() == -1)
    	data.setClassIndex(data.numAttributes() - 1);
    
    // train a G-eRules model and create a set of desciptive rules
    for (Instance instance : Collections.list(data.enumerateInstances())) {

        // this function is here to trigger the real-time adaptaion
        gErules.getVotesForInstance(anInstance)
 
        // add a data instance to a window
        gErules.trainOnInstanceImpl(instance);
         
    }
	
    // get an instance from training set
    Instance anInstance = Collections.list(data.enumerateInstances()).get(10);
    
    // print out all that cover this instance
    for (Rule aRule : gErules.RulesCoveredInstance(anInstance)) {
        System.out.println(aRule.printRule());
    }
    
    // print out all created rules 
    for(Rule aRule: gErules.rulesList){
        System.out.println(aRule.printRule());
    }
}
```

For predictive tasks then the output from _getVotesForInstance()_ returns a double array which contains votes for each class value index. **Note**: class value index is not the acutal value of the class, please refer the structure of an Attribute object from WEKA if you have a difficulity to understand this. 

For describtive tasks, all avaialbe data instances should be used to create a rules library. Once the learning process has completed, the user can inspect all the induced rules as shown in the code example above. 


## Authors
* **Thien Duyen Le**, University of Reading (t.d.le@reading.ac.uk)

Other contributors who participated in this project:

* Frederic Stahl, University of Reading
* Mohamed Medhat Gaber, Birmingham City University
* Joao Gomes, DataRobot, Inc.

## Acknowledgments
This development and the research has been supported by the UK Engineering, and Physical Sciences Research Council (EPSRC) grant EP/M016870/1.