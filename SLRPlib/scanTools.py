"Python code for the c scan tools used in long range phasing"
# examples/increment_example.py
import SLRPlib.tools as tools
import SLRPlib
try:
   from scipy import weave
   from scipy.weave import converters
   import scipy.linalg
   from scipy.weave import ext_tools
except ImportError:
   tools.printerr("Can not find the required scipy library!")
   raise
import numpy


def add_LLscan_and_filter_func(mod):
   "Add combined scaning and filtering function"
   # Type defintions
   peakThreshold = 0.0;
   dipThreshold = 0.0
   cover_limit = 1
   min_length = 10
   from_indiv = 0
   to_indiv = -1
   indPairs = numpy.zeros((1,2),dtype="int")
   genos = numpy.array([[0]],dtype=numpy.int8)
   
   mod.customize.add_support_code("""struct ibd_region_type {
    int i1;
    int i2;
    int bPos;
    int ePos;
    int score;
    bool operator<(const ibd_region_type &B) const {
    bool r=score<B.score
           || (score==B.score && bPos<B.bPos)
           || (score==B.score && bPos==B.bPos && ePos<B.ePos)
           || (score==B.score && bPos==B.bPos && ePos==B.ePos && i1<B.i1)
           || (score==B.score && bPos==B.bPos && ePos==B.ePos && i1==B.i1 && i2<B.i2);
    return r;
    }
};

    """)


   code="#line %d \"scanTools.py\""%(tools.lineno()+1) + """
char *g1, *g2,*g1end;




%(cType)s value,*LLptr,*LLendPtr,cum_value, max_value;
int markers = genos_array->dimensions[1];
int individuals = genos_array->dimensions[0];
%(cType)s _lowLimit, _dipLimit;
_lowLimit = peakThreshold;
_dipLimit = dipThreshold;
const %(cType)s lowLimit = _lowLimit, dipLimit = _dipLimit;


//return_val = 0;
LLendPtr = ((%(cType)s*)LLtable_array->data) + PyArray_NBYTES(LLtable_array)/sizeof(%(cType)s);

PyThreadState *_save;
#define PY_BEGIN_ALLOW_THREADS   _save = PyEval_SaveThread();
#define PY_END_ALLOW_THREADS     PyEval_RestoreThread(_save); 


// Stride information is in bytes but pointer arithmetic in sizeof(type) bytes
const int LLstride = LLtable_array->strides[0]/sizeof(%(cType)s);
const int G1stride = LLtable_array->strides[1]/sizeof(%(cType)s);
const int G2stride = LLtable_array->strides[2]/sizeof(%(cType)s);

std::vector<ibd_region_type> ibd_regions;
std::set<ibd_region_type> out_regions;

/*
kvek_t(int) ibdRegions;
kv_init(ibdRegions);
*/
PY_BEGIN_ALLOW_THREADS;
if(to_indiv<0 || to_indiv>individuals) {
   to_indiv=individuals;
}

for(int i1=from_indiv; i1 < to_indiv ; i1++) {
  for(int i2=0; i2 < individuals ; i2++) {
    if(i1==i2) {
         continue;
    }
    LLptr = (%(cType)s*)LLtable_array->data;
    g1 = (char*)(genos_array->data + genos_array->strides[0]*i1);
    g2 = (char*)(genos_array->data + genos_array->strides[0]*i2);

    g1end = genos_array->data + genos_array->strides[0]*(i1+1);
    cum_value = 0.0,max_value = 0.0;
    int low_idx = 0, peak_idx = 0;
    int idx = 0;
#define USE_DECREASE_LIMIT 1

    while(LLptr < LLendPtr) {
       value = *(LLptr + (*g1)*G1stride + (*g2)*G2stride);
       cum_value += value;


       if( cum_value > max_value) {
           max_value = cum_value;
           peak_idx = idx;
       } else
#ifdef USE_DECREASE_LIMIT
       if( cum_value < (max_value - dipLimit)  || cum_value < 0.0) {
#else
           if( cum_value < 0.0) {
#endif
              if(max_value > lowLimit && (peak_idx-low_idx) >= min_length  ) {  // Gone past a high peak
                  //std::cerr<< "High: ";
                  ibd_region_type high_reg;
                  high_reg.i1 = std::min(i1,i2)*2;
                  high_reg.i2 = std::max(i1,i2)*2;
                  high_reg.bPos = low_idx;
                  high_reg.ePos = peak_idx;
                  high_reg.score=(int)max_value;
                  ibd_regions.push_back(high_reg);
                  std::push_heap(ibd_regions.begin(),ibd_regions.end());

#ifdef USE_DECREASE_LIMIT
              }
              low_idx = peak_idx = idx;
#else
                  // Found a peak. Now find an other one, after that
                  LLptr -= (idx-peak_idx)*LLstride;
                  g1 -= (idx-peak_idx);
                  g2 -= (idx-peak_idx);
                  idx = low_idx = peak_idx;
              } else {
                 // Reset peak tracking
                  low_idx = peak_idx = idx;
              }
#endif
              
              max_value = cum_value = 0.0;
       }
              
       
       idx++;
       LLptr+=LLstride;
       g1++;
       g2++;
    }
    
    if(max_value > lowLimit  && (peak_idx-low_idx) >= min_length ) {  // Gone past a high peak
         ibd_region_type high_reg;
         high_reg.i1 = std::min(i1,i2)*2;
         high_reg.i2 = std::max(i1,i2)*2;
         high_reg.bPos = low_idx;
         high_reg.ePos = peak_idx;
         high_reg.score=(int)max_value;
         ibd_regions.push_back(high_reg);
         std::push_heap(ibd_regions.begin(),ibd_regions.end());
    }
  }
    // Now filter 





// a map from #position to #coverage from here on.
 std::map< int, int >  covers;
std::map<int,int>::iterator i1b,i2b,scan;


#ifndef NIBDFILTERDEBUG
int _covers[markers];
memset(_covers,0,sizeof(int)*markers);
bool brute_high_cover;
int brute_because;
#endif

struct ibd_region_type *cand_ibd,*cand_ibd_end;

int  __count=0;

while(!ibd_regions.empty()) {
    cand_ibd = &ibd_regions.front();
   

   /*
   std::cerr << cand_ibd->i1 <<" "
   << cand_ibd->i2 <<" "
   << cand_ibd->bPos <<" "
   << cand_ibd->ePos <<" "
   << cand_ibd->score << std::endl;
   */






                    

   i1b = covers.lower_bound(cand_ibd->bPos);
   // assert  cand_ibd->bPos <= (*i1b).first || i1b==covers.end()

   // scan until coverage falls below limit 
   bool high_cover = !(  (i1b == covers.end())  ||
                       (i1b == covers.begin() && (*i1b).first>cand_ibd->bPos) );





   if (high_cover) {
       scan = i1b;
       if((*scan).first > cand_ibd->bPos) {
         scan--;
       }
       while(high_cover && scan != covers.end() &&
                        (*scan).first <= cand_ibd->ePos){
            //std::cerr<<(*scan).first<<":"<<(*scan).second<<std::endl;
            //assert(_covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
            if( (*scan).second < cover_limit ) {
                high_cover=false;
            }
            scan++;
       }


   }


   if(!high_cover) {  // Insert this candidate

#ifndef NIBDFILTERDEBUG
      for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
          _covers[cand_ibd->i1/2][i]++;
          _covers[cand_ibd->i2/2][i]++;
      }
#endif

      out_regions.insert(*cand_ibd);


      
      // Increase counts for i1
      if(i1b == covers.end()) {
         i1b = covers.insert(i1b, std::pair<int,int>(cand_ibd->bPos, 1) );
         covers.insert(i1b, std::pair<int,int>(cand_ibd->ePos+1, 0) );
         //assert(_covers[cand_ibd->i1/2][cand_ibd->ePos] == 1);
         //assert(cand_ibd->ePos==7504 || _covers[cand_ibd->i1/2][cand_ibd->ePos+1] == 0);
         //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == 1);
      } else {
         if(cand_ibd->bPos < (*i1b).first ) {
            int coverHere = 0;
            if(i1b!=covers.begin()) {
               scan=i1b;
               scan--;
               coverHere = (*scan).second;
            }
            covers.insert(i1b, std::pair<int,int>(cand_ibd->bPos, coverHere + 1 ));
            //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == i1cover[cand_ibd->bPos]);
         }
         for( scan = i1b; scan != covers.end() &&
                         (*scan).first <= cand_ibd->ePos; scan++){
            (*scan).second++;
            //assert(_covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
         }
         if(scan==covers.end() || (*scan).first > cand_ibd->ePos+1) {
            scan--;
            covers.insert(scan, std::pair<int,int>(cand_ibd->ePos+1, (*scan).second-1) );
            //assert((cand_ibd->ePos==7504 )||_covers[cand_ibd->i1/2][cand_ibd->ePos+1] == (*scan).second -1 );
         }
      }
      //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == i1cover[cand_ibd->bPos]);
      //assert( (cand_ibd->ePos==7504) || _covers[cand_ibd->i1/2][cand_ibd->ePos+1] == i1cover[cand_ibd->ePos+1]);


                    
      assert(!high_cover);
   }
    std::pop_heap(ibd_regions.begin(),ibd_regions.end());ibd_regions.pop_back();


}  
//std::cerr<<"ind1: "<<i1<<" ibdRegions.size() = "<<ibd_regions.size() <<" out_regions.size() = "<<out_regions.size()<< std::endl;




}
PY_END_ALLOW_THREADS;



// Return
npy_intp dims[2] = {out_regions.size(),5};
PyObject *out =  PyArray_SimpleNew(2, dims, PyArray_INT);
PyArrayObject *out_array = (PyArrayObject*)out;
#define OUT2(i,j) (*((int*)(out_array->data + (i)*out_array->strides[0] + (j)*out_array->strides[1])))
PY_BEGIN_ALLOW_THREADS;
int countOut=0;
for(std::set<ibd_region_type>::iterator it=out_regions.begin(); it != out_regions.end(); it++,countOut++ ) {
    OUT2(countOut,0) = (*it).i1;
    OUT2(countOut,1) = (*it).i2;
    OUT2(countOut,2) = (*it).bPos;
    OUT2(countOut,3) = (*it).ePos;
    OUT2(countOut,4) = (*it).score;
}

PY_END_ALLOW_THREADS;
return_val = out;




"""

   mod.customize.add_header("<algorithm>")
   mod.customize.add_header("<set>")
   for cDataType,dataType in (("double",numpy.float64),("float",numpy.float32)):
      LLtable = numpy.zeros((1,3,3),dtype = dataType)
      func = ext_tools.ext_function('LLscan_and_filter_%s'%(cDataType),
                                    code%{"cType":cDataType},
                                    ["from_indiv","to_indiv",
                                     "genos","LLtable", "peakThreshold",
                                     "dipThreshold","cover_limit","min_length"])
      mod.add_function(func)









   

def add_LLscan_func(mod):
    "Add the LLscan function to the module mod"

    # Type defintions
    peakThreshold = 0.0;
    dipThreshold = 0.0
    indPairs = numpy.zeros((1,2),dtype="int")
    genos = numpy.array([[0]],dtype=numpy.int8)
    code="#line %d \"scanTools.py\""%(tools.lineno()+1) + """
char *g1, *g2,*g1end;
int i1,i2;
int total_pairs = indPairs_array->dimensions[0];

%(cType)s value,*LLptr,*LLendPtr,cum_value, max_value;
int markers = genos_array->dimensions[1];
%(cType)s _lowLimit, _dipLimit;
_lowLimit = peakThreshold;
_dipLimit = dipThreshold;
const %(cType)s lowLimit = _lowLimit, dipLimit = _dipLimit;


//return_val = 0;
LLendPtr = ((%(cType)s*)LLtable_array->data) + PyArray_NBYTES(LLtable_array)/sizeof(%(cType)s);

PyThreadState *_save;
#define PY_BEGIN_ALLOW_THREADS   _save = PyEval_SaveThread();
#define PY_END_ALLOW_THREADS     PyEval_RestoreThread(_save); 


// Stride information is in bytes but pointer arithmetic in sizeof(type) bytes
const int LLstride = LLtable_array->strides[0]/sizeof(%(cType)s);
const int G1stride = LLtable_array->strides[1]/sizeof(%(cType)s);
const int G2stride = LLtable_array->strides[2]/sizeof(%(cType)s);

std::vector<int> *ibdRegions = new std::vector<int>();
/*
kvek_t(int) ibdRegions;
kv_init(ibdRegions);
*/
PY_BEGIN_ALLOW_THREADS;

for(int iPair=0; iPair < total_pairs ; iPair++) {
    /* do something with item */
    i1 = INDPAIRS2(iPair,0);
    //indPairs[iPair,0);
    i2 = INDPAIRS2(iPair,1);
    //indPairs(iPair,1);
    

    i1 /= 2;
    i2 /= 2;
    LLptr = (%(cType)s*)LLtable_array->data;
    g1 = (char*)(genos_array->data + genos_array->strides[0]*i1);
    g2 = (char*)(genos_array->data + genos_array->strides[0]*i2);

    g1end = genos_array->data + genos_array->strides[0]*(i1+1);
    cum_value = 0.0,max_value = 0.0;
    int low_idx = 0, peak_idx = 0;
    int idx = 0;
    while(LLptr < LLendPtr) {
       value = *(LLptr + (*g1)*G1stride + (*g2)*G2stride);
       cum_value += value;


       /*
       // Alternative, more straightforward but slower way
       %(cType)s altValue;
       char altg1,altg2;
       altg1=genos(i1,idx);
       altg2=genos(i2,idx);
       altValue = LLtable(idx,(int)altg1,(int)altg2);
       if (PyErr_Occurred()) {
       return_val = 2;  // propagate error 
       goto loopBreak;
    } 
       if(abs(value-altValue)>-0.1) {
           std::cerr<<"err: " << idx <<" "
                    <<value<<" "<<altValue
                    <<" "
                    <<(int)altg1<<(int)altg2
                    <<" xxx "
                    <<(int)*g1<<(int)*g2 <<std::endl;
           //PY_END_ALLOW_THREADS;
           //goto loopBreak;
       } else {
          //std::cerr<<".";
       } 
       // Slow way ending
       */

       if( cum_value > max_value) {
           max_value = cum_value;
           peak_idx = idx;
       } else if( cum_value < (max_value - dipLimit)  || cum_value < 0.0) {
              if(max_value > lowLimit ) {  // Gone past a high peak
                  //std::cerr<< "High: ";
                  ibdRegions->push_back(i1 * 2);
                  ibdRegions->push_back(i2 * 2);
                  ibdRegions->push_back(low_idx);
                  ibdRegions->push_back(peak_idx);
                  ibdRegions->push_back((int)max_value); 

                  /*
                  kv_push(int,ibdRegions,i1 * 2);
                  kv_push(int,ibdRegions,i2 * 2);
                  kv_push(int,ibdRegions,low_idx);
                  kv_push(int,ibdRegions,peak_idx);
                  kv_push(int,ibdRegions,(int)max_value);
                  */
                  
                  //std::cerr<< peak_idx-low_idx << " : " << low_idx << "-" << peak_idx << " " << max_value <<" : " << cum_value <<std::endl;
                  //if(ibdRegions->size() > 20) { goto loopBreak;}
                  //if(peak_idx>700) {goto loopBreak;}
                  //if(kv_size(ibdRegions) > 20) { goto loopBreak;}
              }
              // Reset peak tracking
              low_idx = peak_idx = idx;
              max_value = cum_value = 0.0;
       }
              
       
       idx++;
       LLptr+=LLstride;
       g1++;
       g2++;
    }
    
    if(max_value > lowLimit ) {  // Gone past a high peak
         //std::cerr<< "High: ";
         ibdRegions->push_back(i1 * 2);
         ibdRegions->push_back(i2 * 2);
         ibdRegions->push_back(low_idx);
         ibdRegions->push_back(peak_idx);
         ibdRegions->push_back((int)max_value);
    }

    
}
PY_END_ALLOW_THREADS;

if(ibdRegions->size() > 0 ) {
    npy_intp dims[2] = {ibdRegions->size()/5,5};

    
    PyObject *out =  PyArray_SimpleNew(2, dims, PyArray_INT);
    if(!out) {
       std::cerr<<"Out of memory"<<std::endl;
       }
    memcpy(((PyArrayObject*)out)->data,&(*ibdRegions)[0],sizeof(int)*ibdRegions->size());
    
    return_val = out;
}
//return_val = 1.0;
delete ibdRegions;


"""
    
    for cDataType,dataType in (("double",numpy.float64),("float",numpy.float32)):
        LLtable = numpy.zeros((1,3,3),dtype = dataType)
        func = ext_tools.ext_function('LLscan_%s'%(cDataType),
                                      code%{"cType":cDataType},
                                      ["indPairs","genos","LLtable", "peakThreshold", "dipThreshold"])
        mod.add_function(func)













def add_ibd_region_filter(mod):
    "Add a function to greedily select subset of ibd segments with high coverage"
    

    code="#line %d \"scanTools.py\""%(tools.lineno()+1) + """
const int c_indivs = (int)individuals;
const int c_low_limit = (int)low_limit;
const int candidate_ibd_count = ibd_regions_array->dimensions[0];
bool high_cover;

            PyThreadState *_save;
//#define PY_BEGIN_ALLOW_THREADS   _save = PyEval_SaveThread();
//#define PY_END_ALLOW_THREADS     PyEval_RestoreThread(_save); 

PY_BEGIN_ALLOW_THREADS;



// vector of individuals, each containing a map from #position to #coverage from here on.
std::vector< std::map< int, int > > covers(c_indivs);

std::map<int,int>::iterator i1b,i2b,scan;

std::vector<int> out_regions;

#ifndef NIBDFILTERDEBUG
int _covers[200][7505];
memset(_covers,0,sizeof(int)*200*7505);
bool brute_high_cover;
int brute_because;
#endif

struct ibd_region_type *cand_ibd,*cand_ibd_end;
cand_ibd_end = ((struct ibd_region_type *)ibd_regions_array->data) + candidate_ibd_count;
cand_ibd = (struct ibd_region_type *)ibd_regions_array->data;

int  __count=0;
for(cand_ibd = (struct ibd_region_type *)ibd_regions_array->data; cand_ibd < cand_ibd_end; cand_ibd++,__count++) {
   
//for(cand_ibd = (struct ibd_region_type *)ibd_regions_array->data; cand_ibd < (struct ibd_region_type *)ibd_regions_array->data + 100; cand_ibd++) {
   /*
   std::cerr << cand_ibd->i1 <<" "
   << cand_ibd->i2 <<" "
   << cand_ibd->bPos <<" "
   << cand_ibd->ePos <<" "
   << cand_ibd->score << std::endl;
   */






                    

   std::map<int,int> &i1cover = covers[cand_ibd->i1/2],&i2cover = covers[cand_ibd->i2/2];
   i1b = i1cover.lower_bound(cand_ibd->bPos);
   i2b = i2cover.lower_bound(cand_ibd->bPos);
   // assert  cand_ibd->bPos <= (*i1b).first || i1b==i1cover.end()

   // scan until coverage falls below limit 
   high_cover = !(  (i1b == i1cover.end()) || (i2b == i2cover.end()) ||
                       (i1b == i1cover.begin() && (*i1b).first>cand_ibd->bPos) ||
                       (i2b == i2cover.begin() && (*i2b).first>cand_ibd->bPos) );


#ifndef NIBDFILTERDEBUG
   brute_high_cover=true;
   brute_because=-1;
   for(brute_because=cand_ibd->bPos;brute_high_cover && brute_because<=cand_ibd->ePos;brute_because++ ) {
           assert(brute_because<=cand_ibd->ePos);
           assert(0<=brute_because);
           bool a=(_covers[cand_ibd->i1/2][brute_because]>=c_low_limit);
           bool b= (_covers[cand_ibd->i2/2][brute_because]>=c_low_limit);
           if(!a) {
               assert(brute_because==cand_ibd->bPos || _covers[cand_ibd->i1/2][brute_because] == i1cover[brute_because]);
           }
           if(!b) {
               assert(brute_because==cand_ibd->bPos || _covers[cand_ibd->i2/2][brute_because] == i2cover[brute_because]);
           }
           brute_high_cover&=a;
           brute_high_cover&=b;
           assert(brute_because<=cand_ibd->ePos);
           assert(0<=brute_because);
           //std::cerr<<brute_because<<std::endl;
   }
   brute_because--;
   //std::cerr<<"END:"<<brute_because<<std::endl;
   assert(brute_because<=cand_ibd->ePos);
   assert(0<=brute_because);

#endif

#ifndef NIBDFILTERDEBUG
      {
      for(scan = i1cover.begin(); scan!=i1cover.end();scan++) {
          //std::cerr<<(*scan).first << "=>" << (*scan).second << " = " << _covers[cand_ibd->i1/2][(*scan).first] <<std::endl;
          assert((*scan).first==7505 || _covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
      }
      for(scan = i2cover.begin(); scan!=i2cover.end();scan++) {
          assert((*scan).first==7505 || _covers[cand_ibd->i2/2][(*scan).first] == (*scan).second);
      }
      int prevCount = 0;
      //for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
      for(int i=0; i<=7504;i++) {
          if(_covers[cand_ibd->i1/2][i] != prevCount) {
             //std::cerr<< i<<" -> "<<_covers[cand_ibd->i1/2][i] <<std::endl;
             assert(i1cover.find(i) != i1cover.end());
             prevCount=_covers[cand_ibd->i1/2][i];
          }
          }
      prevCount = 0;
      for(int i=0; i<=7504;i++) {
      //for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
          if(_covers[cand_ibd->i2/2][i] != prevCount) {
             //std::cerr<< i<<" -> "<<_covers[cand_ibd->i1/2][i] <<std::endl;
             assert(i2cover.find(i) != i1cover.end());
             prevCount=_covers[cand_ibd->i2/2][i];
          }
          
      
      }
      }

   assert(high_cover || (!high_cover && !brute_high_cover));
#endif




   if (high_cover) {
       scan = i1b;
       if((*scan).first > cand_ibd->bPos) {
         scan--;
       }
       while(high_cover && scan != i1cover.end() &&
                        (*scan).first <= cand_ibd->ePos){
            //std::cerr<<(*scan).first<<":"<<(*scan).second<<std::endl;
            //assert(_covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
            if( (*scan).second < c_low_limit ) {
                high_cover=false;
            }
            scan++;
       }


       scan = i2b;
       if( (*scan).first > cand_ibd->bPos) {
         scan--;
       }
       while(high_cover && scan != i2cover.end() &&
                        (*scan).first <= cand_ibd->ePos){
            //std::cerr<<(*scan).first<<":"<<(*scan).second<<std::endl;
            //assert(_covers[cand_ibd->i2/2][(*scan).first] == (*scan).second);
            if( (*scan).second < c_low_limit ) {
                high_cover=false;
            }
            scan++;
       }

   }

#ifndef NIBDFILTERDEBUG
   assert(brute_high_cover==high_cover);
#endif

   if(!high_cover) {  // Insert this candidate

#ifndef NIBDFILTERDEBUG
      for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
          _covers[cand_ibd->i1/2][i]++;
          _covers[cand_ibd->i2/2][i]++;
      }
#endif

      out_regions.push_back(cand_ibd->i1);
      out_regions.push_back(cand_ibd->i2);
      out_regions.push_back(cand_ibd->bPos);
      out_regions.push_back(cand_ibd->ePos);
      out_regions.push_back(cand_ibd->score);


      
      // Increase counts for i1
      if(i1b == i1cover.end()) {
         i1b = i1cover.insert(i1b, std::pair<int,int>(cand_ibd->bPos, 1) );
         i1cover.insert(i1b, std::pair<int,int>(cand_ibd->ePos+1, 0) );
         //assert(_covers[cand_ibd->i1/2][cand_ibd->ePos] == 1);
         //assert(cand_ibd->ePos==7504 || _covers[cand_ibd->i1/2][cand_ibd->ePos+1] == 0);
         //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == 1);
      } else {
         if(cand_ibd->bPos < (*i1b).first ) {
            int coverHere = 0;
            if(i1b!=i1cover.begin()) {
               scan=i1b;
               scan--;
               coverHere = (*scan).second;
            }
            i1cover.insert(i1b, std::pair<int,int>(cand_ibd->bPos, coverHere + 1 ));
            //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == i1cover[cand_ibd->bPos]);
         }
         for( scan = i1b; scan != i1cover.end() &&
                         (*scan).first <= cand_ibd->ePos; scan++){
            (*scan).second++;
            //assert(_covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
         }
         if(scan==i1cover.end() || (*scan).first > cand_ibd->ePos+1) {
            scan--;
            i1cover.insert(scan, std::pair<int,int>(cand_ibd->ePos+1, (*scan).second-1) );
            //assert((cand_ibd->ePos==7504 )||_covers[cand_ibd->i1/2][cand_ibd->ePos+1] == (*scan).second -1 );
         }
      }
      //assert(_covers[cand_ibd->i1/2][cand_ibd->bPos] == i1cover[cand_ibd->bPos]);
      //assert( (cand_ibd->ePos==7504) || _covers[cand_ibd->i1/2][cand_ibd->ePos+1] == i1cover[cand_ibd->ePos+1]);


      // Increase counts for i2
      if(i2b == i2cover.end()) {
         i2b = i2cover.insert(i2b, std::pair<int,int>(cand_ibd->bPos, 1) );
         i2cover.insert(i2b, std::pair<int,int>(cand_ibd->ePos+1, 0) );
         //assert(_covers[cand_ibd->i2/2][cand_ibd->ePos] == 1);
         //assert(cand_ibd->ePos==7504 || _covers[cand_ibd->i2/2][cand_ibd->ePos+1] == 0);
         //assert(_covers[cand_ibd->i2/2][cand_ibd->bPos] == 1);
      } else {
         if(cand_ibd->bPos < (*i2b).first ) {
            int coverHere = 0;
            if(i2b!=i2cover.begin()) {
               scan=i2b;
               scan--;
               coverHere = (*scan).second;
            }
            i2cover.insert(i2b, std::pair<int,int>(cand_ibd->bPos, coverHere + 1 ));
            //assert(_covers[cand_ibd->i2/2][cand_ibd->bPos] == i2cover[cand_ibd->bPos]);
         }
         for( scan = i2b; scan != i2cover.end() &&
                         (*scan).first <= cand_ibd->ePos; scan++){
            (*scan).second++;
            //assert(_covers[cand_ibd->i2/2][(*scan).first] == (*scan).second);
         }
         if(scan==i2cover.end() || (*scan).first > cand_ibd->ePos+1) {
            scan--;
            i2cover.insert(scan, std::pair<int,int>(cand_ibd->ePos+1, (*scan).second-1) );
            //assert((cand_ibd->ePos==7504 )||_covers[cand_ibd->i2/2][cand_ibd->ePos+1] == (*scan).second -1 );
         }
      }
#ifndef NIBDFILTERDEBUG
      assert(_covers[cand_ibd->i2/2][cand_ibd->bPos] == i2cover[cand_ibd->bPos]);
      assert( (cand_ibd->ePos==7504) || _covers[cand_ibd->i2/2][cand_ibd->ePos+1] == i2cover[cand_ibd->ePos+1]);



      {
      for(scan = i1cover.begin(); scan!=i1cover.end();scan++) {
          //std::cerr<<(*scan).first << "=>" << (*scan).second << " = " << _covers[cand_ibd->i1/2][(*scan).first] <<std::endl;
          assert((*scan).first==7505 || _covers[cand_ibd->i1/2][(*scan).first] == (*scan).second);
      }
      for(scan = i2cover.begin(); scan!=i2cover.end();scan++) {
          assert((*scan).first==7505 || _covers[cand_ibd->i2/2][(*scan).first] == (*scan).second);
      }
      int prevCount = 0;
      //for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
      for(int i=0; i<=7504;i++) {
          if(_covers[cand_ibd->i1/2][i] != prevCount) {
             //std::cerr<< i<<" -> "<<_covers[cand_ibd->i1/2][i] <<std::endl;
             assert(i1cover.find(i) != i1cover.end());
             prevCount=_covers[cand_ibd->i1/2][i];
          }
          }
      prevCount = 0;
      for(int i=0; i<=7504;i++) {
      //for(int i=cand_ibd->bPos; i<=cand_ibd->ePos;i++) {
          if(_covers[cand_ibd->i2/2][i] != prevCount) {
             //std::cerr<< i<<" -> "<<_covers[cand_ibd->i1/2][i] <<std::endl;
             assert(i2cover.find(i) != i1cover.end());
             prevCount=_covers[cand_ibd->i2/2][i];
          }
          
      
      }
      }

#endif

                    
      assert(!high_cover);
   }


}
std::cerr<<std::endl;

PY_END_ALLOW_THREADS;

// Return
npy_intp dims[2] = {out_regions.size()/5,5};
PyObject *out =  PyArray_SimpleNew(2, dims, PyArray_INT);
memcpy(((PyArrayObject*)out)->data,&out_regions[0],sizeof(int)*out_regions.size());

return_val = out;


"""

    ibd_regions = numpy.zeros((1,5),dtype=numpy.int32)
    low_limit = 1
    individuals = 1


    func = ext_tools.ext_function('IBD_filter_c',  code,
                                  ["ibd_regions","low_limit","individuals"])
    mod.add_function(func)



def add_processAllocedIBD(mod):
    """Add the function to do the phasing proper, without MPI"""

    code="#line %d \"scanTools.py\""%(tools.lineno()+1) + """

//#ifndef NDEBUG
    std::cout<<"function: "<<__func__<<std::endl;
//#endif

    int ij,j;
    int h0,h1,p,p1;
    const int ibd_chunks = Nind1firstHaplo[0];
    const int num_markers = NhLike[1];
    const int num_indivs = NhLike[0];
    %(cType)s old_p2h[2][4];
    double sum_sq_err,tot_sum_sq_err = 0.0;
    unsigned int tot_markers_err = 0;
    
    %(cType)s p2p[num_markers+1][5];
    %(cType)s p2pP[num_markers+1][5];

    // Trickery needed to send array of objects to ext_tools
#undef P2H12
#undef P2H22
    //PyArrayObject *p2h1_array_idx = convert_to_numpy((PyObject*)P2H11(0), "p2h1(0)");
    //conversion_numpy_check_type( p2h1_array_idx ,%(pyCtype)s, "p2h1(0)");
    //p2h1_array_idx = convert_to_numpy((PyObject*)P2H11(ibd_chunks-1), "p2h1(ibd_chunks-1)");

    //conversion_numpy_check_type( p2h1_array_idx ,%(pyCtype)s, "p2h1(ibd_chunks-1)");
    
    //conversion_numpy_check_type( (PyArrayObject*)P2H11(ibd_chunks-1) ,%(pyCtype)s, "p2h1");
    //conversion_numpy_check_type( (PyArrayObject*)P2H21(0) ,%(pyCtype)s, "p2h2");
    //conversion_numpy_check_type( (PyArrayObject*)P2H21(ibd_chunks-1) ,%(pyCtype)s, "p2h2");

#undef CPT
#define CPT(marker,pPrev,pNext,h0,h1) (*((%(cType)s*)(CPT_array->data + (marker)*SCPT[0] + (pPrev)*SCPT[1] + (pNext)*SCPT[2] + (h0)*SCPT[3] + (h1)*SCPT[4]   )))
    PyThreadState *_save;

    Py_BEGIN_ALLOW_THREADS;

    for(int ibd_idx = 0; ibd_idx < ibd_chunks; ibd_idx++) {
        const int endM = ENDMARKER1(ibd_idx);
        const int beginM = BEGINMARKER1(ibd_idx);
        const int ind1 = IND1FIRSTHAPLO1(ibd_idx)/2;
        const int ind2 = IND2FIRSTHAPLO1(ibd_idx)/2;



        
        if( 0 && PREVMEANSQRDIFF3(ibd_idx,0,0) < 2.3e-16 ) {
            tot_sum_sq_err += PREVMEANSQRDIFF3(ibd_idx,0,0);
            PREVMEANSQRDIFF3(ibd_idx,0,0) *= 2;
            std::cout<<"Skipping "<<ibd_idx<<std::endl;
            continue;
        }
        
        // Trickery needed to send array of objects to ext_tools
        PyArrayObject* p2h1_array_ind = (PyArrayObject*)P2H11(ibd_idx);
        PyArrayObject* p2h2_array_ind = (PyArrayObject*)P2H21(ibd_idx);

#define P2H1(i,j)  (*((%(cType)s*)(p2h1_array_ind->data + (i)*p2h1_array_ind->strides[0] + (j)*p2h1_array_ind->strides[1])))
#define P2H2(i,j)  (*((%(cType)s*)(p2h2_array_ind->data + (i)*p2h2_array_ind->strides[0] + (j)*p2h2_array_ind->strides[1])))
#ifndef NDEBUG
        //PY_END_ALLOW_THREADS;
        if(!PyArray_Check((PyObject*)p2h1_array_ind)) {
           handle_bad_type((PyObject*)p2h1_array_ind,"numpy","P2H1(ibd_idx)");
        }
        conversion_numpy_check_type( p2h1_array_ind ,%(pyCtype)s, "p2h1_array_ind"); 

        if(!PyArray_Check((PyObject*)p2h2_array_ind)) {
           handle_bad_type((PyObject*)p2h2_array_ind,"numpy","P2H2(ibd_idx)");
        }
        conversion_numpy_check_type( p2h2_array_ind ,%(pyCtype)s, "p2h2_array_ind"); 
        //PY_BEGIN_ALLOW_THREADS;
#endif
        sum_sq_err = 0.0;
        
        // Forward
        
        // First step

        for(int p=0;p<5;p++) {
            if(beginM == 0 ) {
                p2p[0][p] = (%(cType)s)FIRSTCP2P1(p);
                //p2p[0][p] = 0.0;
            } else {
                if(p == 4) { // Due to overhang, we always start from non ibd state
                   p2p[0][p] = 0.0;
                } else {
                   p2p[0][p] = -log(0.0);
                }
            }
        }

        // Rest of the steps
        for(int ij=0,j = beginM ; j <= endM;j++,ij++) {
             %(cType)s tot_min_p2p = 1e308;

             // p_j -> p_{j+1}

             for(int p=0;p<5;p++) {
                 %(cType)s  p2p_V = 1e307, min_p2p = 1e308;
                    
                  for(int p0=0; p0<5; p0++) {
                      // From previous p to current p
                       const %(cType)s p2p0_V =  p2p[ij][p0];
                       for(int h0=0;h0<4;h0++) {
                         // From diplotype 1 to p
                         const %(cType)s h12p_V =  HLIKE3(ind1,j,h0) - P2H1(ij,h0);  //h12ca(ij,h0);
                         for(h1=0;h1<4;h1++) {
                            // From diplotype 2 to p
                            const %(cType)s h22p_V =  HLIKE3(ind2,j,h1) - P2H2(ij,h1);  //h22ca(ij,h1);
                            const %(cType)s CPT_V = CPT(j, p0 ,p,h0,h1) ;
                            assert(CPT_V>=0.0);
                            assert(h22p_V>=0.0);
                            //std::cerr<<ca2p_V<<"->";
                            //ca2p_V = std::min(ca2p_V, CPT(j, p0 ,p,h0,h1) + h12ca_V + h22ca(ij,h1) + ca2p0_V );
                            //p2p_V = std::min(p2p_V, CPT(j, p0 ,p,h0,h1) + h12p_V + h22p_V + p2p0_V );
                            p2p_V = std::min(p2p_V, CPT(j, p0 ,p,h0,h1) + h12p_V + h22p_V + p2p0_V );
                            assert( p2p_V >= 0.0);
                         }

                       }
                       if(min_p2p > p2p_V ) {
                          p2p[ij+1][p] = p2p_V;
                          min_p2p = p2p_V;
                       }

                    }
                    tot_min_p2p = std::min(tot_min_p2p,min_p2p);
                 }
                 // Normalize
                  //std::cout<<"p2p["<<ij+1<< "]={";
                 for(int p=0;p<5;p++) { 
                    p2p[ij+1][p] -= tot_min_p2p;
                    //std::cout<<p2p[ij+1][p]<<",";
                 }
                 //std::cout<<"}"<<std::endl;

                 //std::cout<<ij<<": tot_min_p2p="<< tot_min_p2p<<std::endl;
            }



            // Backward
            
            for(int p=0;p<5;p++) { //Initialization
               if(endM == (num_markers - 1) ) { // No information past the end of chromosome
                  //ca2pP(p,endM-beginM+1)=(%(cType)s)firstCP2P(p); // THIS IS UNDER TESTING #  Fri Sep 03 16:35:32 BST 2010 
                  //p2pP[endM-beginM+1][p]=0.0;
                  p2pP[endM-beginM+1][p]=(%(cType)s)FIRSTCP2P1(p);
                  //std::cerr<<"End of chromosome@"<<endM<<": ca2pP <- " << ca2pP(p,endM-beginM+1) <<std::endl;
               } else { // Within the chromosome, IBD region ends in non IBD state (due to overhang)
                  if(p == 4) {  
                     p2pP[endM-beginM+1][p] = 0.0;
                  } else {
                     p2pP[endM-beginM+1][p] = (%(cType)s)-log(0.0);
                  }
                  //std::cerr<<"ca2pP <- " << ca2pP(p,endM-beginM+1) <<std::endl;
               } 
            }
            for(int ij=endM-beginM,j=endM; j>=beginM; j--,ij--) {
                // Backwards from P(p_j| h0,h1,p_j-1) (to h0, h1, and p_j-1]
                %(cType)s min_p2pP[5] = {1e308,1e308,1e308,1e308,1e308};
                %(cType)s tmp_p2h1[4] = {1e308,1e308,1e308,1e308};
                %(cType)s tmp_p2h2[4] = {1e308,1e308,1e308,1e308};

                for(int p=0;p<5;p++) {
                    const %(cType)s p2pP_V = p2pP[ij+1][p];
                    for(int p1=0;p1<5;p1++) {
                       const %(cType)s p2p_V = p2p[ij+1][p1];
                       for(int h0=0;h0<4;h0++) {
                           const %(cType)s h12p_V =  HLIKE3(ind1,j,h0) - P2H1(ij,h0);  //h12ca(ij,h0);
                           for(int h1=0;h1<4;h1++) {
                               const %(cType)s CPT_V =  CPT(j,p1,p,h0,h1); 
                               const %(cType)s h22p_V =  HLIKE3(ind2,j,h1) - P2H2(ij,h1);  //h22ca(ij,h1);
                               
                               min_p2pP[p1] = std::min(min_p2pP[p1], CPT_V +         p2pP_V + h12p_V + h22p_V );
                               tmp_p2h1[h0] = std::min(tmp_p2h1[h0], CPT_V + p2p_V + p2pP_V +          h22p_V );
                               tmp_p2h2[h1] = std::min(tmp_p2h2[h1], CPT_V + p2p_V + p2pP_V + h12p_V );
                           }
                       }
                   }
                }
                // Normalize and set:
                %(cType)s minVal=1e308;
                for(int p=0;p<5;p++) {
                     minVal=std::min(minVal, min_p2pP[p]);
                }
                //std::cout<<ij<<": minVal="<< minVal<<std::endl;
                for(int p=0;p<5;p++) {
                     p2pP[ij][p] = min_p2pP[p] - minVal;
                }

                

                minVal = 1e308;
                %(cType)s minVal2 = 1e308; 
                for(int h0=0;h0<4;h0++) {
                    old_p2h[0][h0] = P2H1(ij,h0);
                    old_p2h[1][h0] = P2H2(ij,h0);

                    P2H1(ij,h0) = dampF * P2H1(ij,h0) + (1-dampF)*tmp_p2h1[h0];
                    P2H2(ij,h0) = dampF * P2H2(ij,h0) + (1-dampF)*tmp_p2h2[h0];


                    minVal=std::min(minVal, P2H1(ij,h0));
                    minVal2=std::min(minVal2, P2H2(ij,h0));
                }
                for(int h0=0;h0<4;h0++) {
                    P2H1(ij,h0) -= minVal;
                    P2H2(ij,h0) -= minVal2;

                    // Count error
                    double err_value ;
                    err_value = (old_p2h[0][h0] - P2H1(ij,h0));
                    sum_sq_err += err_value * err_value;
                    err_value = (old_p2h[1][h0] - P2H2(ij,h0));
                    sum_sq_err += err_value * err_value;

                    HLIKE3(ind1,j,h0) += P2H1(ij,h0) - old_p2h[0][h0];
                    HLIKE3(ind2,j,h0) += P2H2(ij,h0) - old_p2h[1][h0];
                }


            }

            PREVMEANSQRDIFF3(ibd_idx,0,0) = sum_sq_err / ( endM - beginM + 1) ;
            tot_sum_sq_err += sum_sq_err / ( endM - beginM + 1) ;
            //tot_markers_err += 


    }

    Py_END_ALLOW_THREADS;
            
    return_val = tot_sum_sq_err ;
#undef P2H1
#undef P2H2


    """
    from scipy.weave.c_spec import num_to_c_types
    from scipy.weave.standard_array_spec import num_typecode
    num_to_c_types["O"] = "PyArrayObject*"
    num_typecode["O"] = "PyArray_OBJECT"

    import pdb
    #pdb.set_trace()
    for cDataType,pyCtype,dataType in (("double","PyArray_DOUBLE",numpy.float64),("float","PyArray_FLOAT",numpy.float32)):
        allocedIBD_dtype = { 'names' : ( "ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "lastMarkerFilled", "p2h1", "p2h2",  "prevMeanSqrDiff", "endMarker"),
                             'formats' : (numpy.int32,     numpy.int32,       numpy.int32,    numpy.int32,   numpy.object, numpy.object, (numpy.float64,(1,1)), numpy.int32) }

        #allocedIBD_dtype = { 'names' : ( "ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "lastMarkerFilled", "p2h1", "p2h2",  "prevMeanSqrDiff", "endMarker"),
        #                     'formats' : (numpy.int32,     numpy.int32,       numpy.int32,    numpy.int32,   (dataType,(5,4)), (dataType,(5,4)), (numpy.float64,(1,1)), numpy.int32) }
        allocedIBD=numpy.empty(2,dtype = allocedIBD_dtype)

        dampF = 0.0
        firstCP2P = numpy.empty(5, dtype = dataType)
        CPT = numpy.empty((6,5,5,4,4), dtype= dataType)
        ind1firstHaplo = allocedIBD["ind1firstHaplo"]
        ind2firstHaplo = allocedIBD["ind2firstHaplo"]
        beginMarker =  allocedIBD["beginMarker"]
        endMarker =  allocedIBD["endMarker"]
        lastMarkerFilled =  allocedIBD["lastMarkerFilled"]
        p2h1 =  allocedIBD["p2h1"]
        p2h2 =  allocedIBD["p2h2"]
        prevMeanSqrDiff =  allocedIBD["prevMeanSqrDiff"]



        # hLike.shape = (individuals, markers, 4)
        hLike = numpy.zeros((1,2,4),dtype=dataType)
        func = ext_tools.ext_function('_processAllocedIBD_%s'%(cDataType),
                                      code%{"cType":cDataType,"pyCtype":pyCtype},
                                      ["ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "endMarker", "p2h1", "p2h2",  "prevMeanSqrDiff", "hLike","CPT","dampF","firstCP2P"])
        mod.add_function(func)












def add_processAllocedIBD_sumProduct(mod):
    """Add the function to do the phasing proper, without MPI. Experimental Sum-product version (instead of min-sum)"""

    code="#line %d \"scanTools.py\""%(tools.lineno()+1) + """

#ifndef NDEBUG
    std::cout<<"function: "<<__func__<<std::endl;
#endif
    
    int ij,j;
    int h0,h1,p,p1;
    const int ibd_chunks = Nind1firstHaplo[0];
    const int num_markers = NhLike[1];
    const int num_indivs = NhLike[0];
    %(cType)s old_p2h[2][4];
    double sum_sq_err,tot_sum_sq_err = 0.0;
    unsigned int tot_markers_err = 0;
    
    %(cType)s p2p[num_markers+1][5];
    %(cType)s p2pP[num_markers+1][5];

    // Trickery needed to send array of objects to ext_tools
#undef P2H12
#undef P2H22
    //conversion_numpy_check_type( (PyArrayObject*)P2H11(0) ,%(pyCtype)s, "p2h1");
    //conversion_numpy_check_type( (PyArrayObject*)P2H11(ibd_chunks-1) ,%(pyCtype)s, "p2h1");
    //conversion_numpy_check_type( (PyArrayObject*)P2H21(0) ,%(pyCtype)s, "p2h2");
    //conversion_numpy_check_type( (PyArrayObject*)P2H21(ibd_chunks-1) ,%(pyCtype)s, "p2h2");

    typedef %(cType)s (*CPT_ptr_t)[5][5][4][4];
    typedef %(cType)s (*hLike_ptr_t)[4];


#undef CPT    
#define CPT(marker,pPrev,pNext,h0,h1) (*((%(cType)s*)(CPT_array->data + (marker)*SCPT[0] + (pPrev)*SCPT[1] + (pNext)*SCPT[2] + (h0)*SCPT[3] + (h1)*SCPT[4]   )))

    Py_BEGIN_ALLOW_THREADS;

    for(int ibd_idx = 0; ibd_idx < ibd_chunks; ibd_idx++) {
        const int endM = ENDMARKER1(ibd_idx);
        const int beginM = BEGINMARKER1(ibd_idx);
        const int ind1 = IND1FIRSTHAPLO1(ibd_idx)/2;
        const int ind2 = IND2FIRSTHAPLO1(ibd_idx)/2;

        // Trickery needed to send array of objects to ext_tools
        PyArrayObject * p2h1_array_ind = (PyArrayObject*)P2H11(ibd_idx);
        PyArrayObject * p2h2_array_ind = (PyArrayObject*)P2H21(ibd_idx);

#define P2H1(i,j)  (*((%(cType)s*)(p2h1_array_ind->data + (i)*p2h1_array_ind->strides[0] + (j)*p2h1_array_ind->strides[1])))
#define P2H2(i,j)  (*((%(cType)s*)(p2h2_array_ind->data + (i)*p2h2_array_ind->strides[0] + (j)*p2h2_array_ind->strides[1])))
//        hLike_ptr_t __restrict__ p2h1_ptr = (hLike_ptr_t)(p2h1_array->data);
//        hLike_ptr_t __restrict__ p2h2_ptr = (hLike_ptr_t)(p2h2_array->data);
//#define P2H1(pos,h) (p2h1_ptr[pos])[h]
//#define P2H2(pos,h) (p2h2_ptr[pos])[h]

#ifndef NDEBUG
       conversion_numpy_check_type( p2h1_array_ind ,%(pyCtype)s, "p2h1_array_ind"); 
       conversion_numpy_check_type( p2h2_array_ind ,%(pyCtype)s, "p2h2_array_ind"); 
#endif
        sum_sq_err = 0.0;

        
        // Forward
        
        // First step

        for(int p=0;p<5;p++) {
            if(beginM == 0 ) {
                p2p[0][p] = 0.2;
            } else {
                if(p == 4) {
                   p2p[0][p] = 1.0-0.001*4;
                } else {
                   p2p[0][p] = 0.001;
                }
            }
        }

        // Rest of the steps
        for(int ij=0,j = beginM ; j <= endM;j++,ij++) {
             %(cType)s tot_sum_p2p = 0.0;

//             const CPT_ptr_t __restrict__ CPT_ptr = (CPT_ptr_t)(CPT_array->data + CPT_array->strides[0]*j);
//             const hLike_ptr_t __restrict__ hLike1_ptr = (hLike_ptr_t)(hLike_array->data + hLike_array->strides[0]*ind1 + hLike_array->strides[1]*j);
//             const hLike_ptr_t __restrict__ hLike2_ptr = (hLike_ptr_t)(hLike_array->data + hLike_array->strides[0]*ind2 + hLike_array->strides[1]*j);

                                                                    


             // p_j -> p_{j+1}

             for(int p=0;p<5;p++) {
                 %(cType)s  p2p_V = 0.0;
                    
                  for(int p0=0; p0<5; p0++) {
                      // From previous p to current p
                       const %(cType)s p2p0_V =  p2p[ij][p0];
                       for(int h0=0;h0<4;h0++) {
                         // From diplotype 1 to p
                         const %(cType)s h12p_V =  HLIKE3(ind1,j,h0) / P2H1(ij,h0);  //h12ca(ij,h0);
                         assert(!isnan(h12p_V));
                         for(h1=0;h1<4;h1++) {
                            // From diplotype 2 to p
                            const %(cType)s h22p_V =  HLIKE3(ind2,j,h1) / P2H2(ij,h1);  //h22ca(ij,h1);
                            assert(!isnan(h22p_V));
                            //std::cerr<<ca2p_V<<"->";
                            //ca2p_V = std::min(ca2p_V, CPT(j, p0 ,p,h0,h1) + h12ca_V + h22ca(ij,h1) + ca2p0_V );
                            assert(!isnan(CPT(j, p0 ,p,h0,h1)));
                            //double CPT_V = CPT(j, p0 ,p,h0,h1);
                            //assert( (*CPT_ptr)[p0][p][h0][h1] == CPT_V);
                            p2p_V += CPT(j, p0 ,p,h0,h1) * h12p_V * h22p_V * p2p0_V;
                            //p2p_V += (*CPT_ptr)[p0][p][h0][h1]* h12p_V * h22p_V * p2p0_V;
                            assert(!isnan(p2p_V));

                         }

                       }
                  }
                  p2p[ij+1][p] = p2p_V;
                  assert(!isnan(p2p_V));

                  tot_sum_p2p += p2p_V;
                  assert(!isnan(tot_sum_p2p));
             }
             // Normalize
             for(int p=0;p<5;p++) {
                p2p[ij+1][p] /= tot_sum_p2p;
                assert(!isnan(p2p[ij+1][p]));
             }
        }





            // Backward
            
            for(int p=0;p<5;p++) { //Initialization
               if(endM == (num_markers - 1) ) { // No information past the end of chromosome
                  p2pP[endM-beginM+1][p]=0.2;
               } else { // Within the chromosome, IBD region ends in non IBD state
                  if(p == 4) {
                     p2pP[endM-beginM+1][p] = 1.0 - 0.001*4;
                  } else {
                     p2pP[endM-beginM+1][p] = 0.001; 
                  }
                  //std::cerr<<"ca2pP <- " << ca2pP(p,endM-beginM+1) <<std::endl;
               } 
            }
            for(int ij=endM-beginM,j=endM; j>=beginM; j--,ij--) {
                // Backwards from P(p_j| h0,h1,p_j-1) (to h0, h1, and p_j-1]
                %(cType)s tmp_p2pP[5] = {0.0,0.0,0.0,0.0,0.0};
                %(cType)s tmp_p2h1[4] = {0.0,0.0,0.0,0.0};
                %(cType)s tmp_p2h2[4] = {0.0,0.0,0.0,0.0};

                const CPT_ptr_t __restrict__ CPT_ptr = (CPT_ptr_t)(CPT_array->data + CPT_array->strides[0]*j);


                for(int p=0;p<5;p++) {
                    const %(cType)s p2pP_V = p2pP[ij+1][p];
                    for(int p1=0;p1<5;p1++) {
                       const %(cType)s p2p_V = p2p[ij+1][p1];
                       for(int h0=0;h0<4;h0++) {
                           const %(cType)s h12p_V =  HLIKE3(ind1,j,h0) / P2H1(ij,h0);  //h12ca(ij,h0);
                           for(int h1=0;h1<4;h1++) {
                               //const %(cType)s CPT_V =  CPT(j,p1,p,h0,h1);
                               const %(cType)s CPT_V =  (*CPT_ptr)[p1][p][h0][h1]; 
                               const %(cType)s h22p_V =  HLIKE3(ind2,j,h1) / P2H2(ij,h1);  //h22ca(ij,h1);
                               
                               tmp_p2pP[p1] +=  CPT_V *         p2pP_V * h12p_V * h22p_V ;
                               tmp_p2h1[h0] +=  CPT_V * p2p_V * p2pP_V *          h22p_V ;
                               tmp_p2h2[h1] +=  CPT_V * p2p_V * p2pP_V * h12p_V ;
                           }
                       }
                   }
                }
                // Normalize and set:
                %(cType)s sumVal=0.0;
                for(int p=0;p<5;p++) {
                     sumVal += tmp_p2pP[p];
                }
                for(int p=0;p<5;p++) {
                     p2pP[ij][p] = tmp_p2pP[p]/ sumVal;
                }

                

                sumVal = 0.0;
                %(cType)s sumVal2 = 0.0; 
                for(int h0=0;h0<4;h0++) {
                    old_p2h[0][h0] = P2H1(ij,h0);
                    old_p2h[1][h0] = P2H2(ij,h0);

                    P2H1(ij,h0) = dampF * P2H1(ij,h0) + (1-dampF)*tmp_p2h1[h0];
                    P2H2(ij,h0) = dampF * P2H2(ij,h0) + (1-dampF)*tmp_p2h2[h0];
 

                    sumVal  += P2H1(ij,h0);
                    sumVal2 += P2H2(ij,h0);
                }
                for(int h0=0;h0<4;h0++) {
                    P2H1(ij,h0) /= sumVal;
                    P2H2(ij,h0) /= sumVal2;

                    // Count error
                    double err_value ;
                    err_value = (old_p2h[0][h0] - P2H1(ij,h0));
                    sum_sq_err += err_value * err_value;
                    err_value = (old_p2h[1][h0] - P2H2(ij,h0));
                    sum_sq_err += err_value * err_value;

                    HLIKE3(ind1,j,h0) *= P2H1(ij,h0) / old_p2h[0][h0];
                    HLIKE3(ind2,j,h0) *= P2H2(ij,h0) / old_p2h[1][h0];

                    assert(!isnan(HLIKE3(ind1,j,h0)));
                    assert(!isnan(HLIKE3(ind2,j,h0)));
                }


            }
            assert(isnan(tot_sum_sq_err) == 0);
            tot_sum_sq_err += sum_sq_err; 
            tot_markers_err += ( endM - beginM + 1) ;
            assert(isnan(tot_sum_sq_err) == 0);


    }



    Py_END_ALLOW_THREADS;
    assert(!isnan(tot_sum_sq_err));
    assert(!isnan(tot_markers_err));
    assert(isnan(tot_sum_sq_err)==0);
    return_val = tot_sum_sq_err / tot_markers_err;

#undef P2H1
#undef P2H2

    """
    from scipy.weave.c_spec import num_to_c_types
    from scipy.weave.standard_array_spec import num_typecode
    num_to_c_types["O"] = "PyArrayObject*"
    num_typecode["O"] = "PyArray_OBJECT"

    import pdb
    #pdb.set_trace()
    for cDataType,pyCtype,dataType in (("double","PyArray_DOUBLE",numpy.float64),("float","PyArray_FLOAT",numpy.float32)):
        allocedIBD_dtype = { 'names' : ( "ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "lastMarkerFilled", "p2h1", "p2h2",  "prevMeanSqrDiff", "endMarker"),
                             'formats' : (numpy.int32,     numpy.int32,       numpy.int32,    numpy.int32,   numpy.object, numpy.object, (numpy.float64,(1,1)), numpy.int32) }

        #allocedIBD_dtype = { 'names' : ( "ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "lastMarkerFilled", "p2h1", "p2h2",  "prevMeanSqrDiff", "endMarker"),
        #                     'formats' : (numpy.int32,     numpy.int32,       numpy.int32,    numpy.int32,   (dataType,(5,4)), (dataType,(5,4)), (numpy.float64,(1,1)), numpy.int32) }
        allocedIBD=numpy.empty(2,dtype = allocedIBD_dtype)

        dampF = 0.0
        firstCP2P = numpy.empty(5, dtype = dataType)
        CPT = numpy.empty((6,5,5,4,4), dtype= dataType)
        ind1firstHaplo = allocedIBD["ind1firstHaplo"]
        ind2firstHaplo = allocedIBD["ind2firstHaplo"]
        beginMarker =  allocedIBD["beginMarker"]
        endMarker =  allocedIBD["endMarker"]
        lastMarkerFilled =  allocedIBD["lastMarkerFilled"]
        p2h1 =  allocedIBD["p2h1"]
        p2h2 =  allocedIBD["p2h2"]
        prevMeanSqrDiff =  allocedIBD["prevMeanSqrDiff"]

        # hLike.shape = (individuals, markers, 4)
        hLike = numpy.zeros((1,2,4),dtype=dataType)
        func = ext_tools.ext_function('_processAllocedIBD_sumProduct_%s'%(cDataType),
                                      code%{"cType":cDataType,"pyCtype":pyCtype},
                                      ["ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "endMarker", "p2h1", "p2h2",  "prevMeanSqrDiff", "hLike","CPT","dampF","firstCP2P"])
        mod.add_function(func)




   

def build_c_scan_ext():
    """ Build the C extensions needed in longRangePhasing class.
        The extension will be built in the local directory.
    """
    import pdb
    import os.path
    #print "My file name:",__file__
    mod = ext_tools.ext_module('c_scan_ext',compiler="gcc")

    mod.customize.add_header("<map>")
    mod.customize.add_header("<vector>")
    mod.customize.add_header("<cmath>")
    mod.customize.add_header("<assert.h>")

    add_LLscan_and_filter_func(mod)
    add_LLscan_func(mod)
    add_ibd_region_filter(mod)
    add_processAllocedIBD(mod)
    add_processAllocedIBD_sumProduct(mod)

    #pdb.set_trace()
    mod.customize.add_extra_compile_arg("-g")
    mod.customize.add_extra_compile_arg("-Wall")
    #mod.customize.add_extra_compile_arg("-O3")
    #mod.customize.add_extra_compile_arg("-ftree-vectorizer-verbose=3")
    mod.customize.add_extra_compile_arg("-DNIBDFILTERDEBUG")
    mod.customize.add_extra_compile_arg("-DNDEBUG")
    mod.compile(verbose=2,location = os.path.dirname(__file__))

try:
    # FOLLOWING LINE FOR TESTING.
    build_c_scan_ext()
    import c_scan_ext
except ImportError:
    build_c_scan_ext()
    import c_scan_ext


class c_ext:
    def __init__(self,dtype):
        self.dtype = dtype
        if self.dtype == numpy.float64:
            self.cDtype = "double"
        elif self.dtype == numpy.float32:
            self.cDtype = "float"
        else:
            raise Exception()

    
    def __getattr__(self,attr):
        return getattr(c_scan_ext,attr+"_" + self.cDtype)
    

    def IBD_filter(self,ibd_regions,individuals,ibd_limit,ibd_length_limit):
        "Return subset of ibd_regions such that all (most) sites have coverage at least ibd_limit"
        assert ibd_regions.flags.c_contiguous, "ibd_regions must be c-contiguous"
        assert len(ibd_regions.shape) == 2, "ibd_regions must be two dimensional, with 5 columns"
        assert ibd_regions.shape[1] == 5, "ibd_regions must have row structure [ind1, ind2, beginMarker, endMarker,score]"
        #pdb.set_trace()
        ibd_regions = ibd_regions[(ibd_regions[:,3]-ibd_regions[:,2]) >= ibd_length_limit]
        ibd_regions = tools.sortDescendingByCol(ibd_regions,4)
        #ibd_regions = ibd_regions[:17,:]

        print ibd_regions

        #print ibd_regions
        return c_scan_ext.IBD_filter_c(ibd_regions,ibd_limit,individuals)

    def processAllocedIBD(self,allocedIBD,hLike,CPT,dampF,firstCP2P,MAPestimate=True):
        assert len(allocedIBD) > 0, "Must have something to process"

        ind1 = allocedIBD["ind1firstHaplo"]
        ind2 = allocedIBD["ind2firstHaplo"]
        beginMarker =  allocedIBD["beginMarker"]
        endMarker =  allocedIBD["lastMarkerFilled"]  # For our purposes, this is the end marker
        p2h1 =  allocedIBD["p2h1"]
        p2h2 =  allocedIBD["p2h2"]
        prevMeanSqrDiff =  allocedIBD["prevMeanSqrDiff"]

        import pdb
        #pdb.set_trace()
        #assert CPT.flags.c_contiguous, "CPT must be c-contiguous"
        assert hLike.flags.c_contiguous, "CPT must be c-contiguous"
        if MAPestimate:
            return self._processAllocedIBD(ind1, ind2,  beginMarker,  endMarker, p2h1, p2h2,  prevMeanSqrDiff,hLike,CPT,dampF,firstCP2P)
        else:
            return self._processAllocedIBD_sumProduct(ind1, ind2,  beginMarker,  endMarker, p2h1, p2h2,  prevMeanSqrDiff,hLike,CPT,dampF,firstCP2P)
