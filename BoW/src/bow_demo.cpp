#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3/DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "DBoW3/DescManip.h"

using namespace DBoW3;
using namespace std;


// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}




vector< cv::Mat  >  loadFeatures() 
{
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    fdetector=cv::ORB::create();
    vector<cv::Mat>  features;
    cout << "Extracting  features..." << endl;
    for(size_t i = 0; i < 10; ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
	string path="../data/img/"+to_string(i+1)+".png";
        
	cout<<"reading image: "<<path<<endl;
        cv::Mat image = cv::imread(path,0);
        
	cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        
	cout<<"done detecting features"<<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(size_t i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for(size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("../data/voc/small_voc.yml.gz");
    cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("../data/voc/small_voc.yml.gz");
    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now
    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

   /* // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;*/
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{
    vector< cv::Mat> features= loadFeatures();
    
    testVocCreation(features);

    testDatabase(features);

    return 0;
}