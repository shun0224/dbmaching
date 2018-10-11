package jp.jobdirect.dbmatching.app;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import com.worksap.nlp.sudachi.Dictionary;
import com.worksap.nlp.sudachi.DictionaryFactory;
import com.worksap.nlp.sudachi.Morpheme;
import com.worksap.nlp.sudachi.Tokenizer;

import jp.jobdirect.dbmatching.classifier.AbstractClassifier;
import jp.jobdirect.dbmatching.classifier.MatchWithLikelihood;
import jp.jobdirect.dbmatching.classifier.WeakClassifier;
import jp.jobdirect.dbmatching.model.DataSet;
import jp.jobdirect.dbmatching.model.Match;
import jp.jobdirect.dbmatching.model.Record;
import jp.jobdirect.dbmatching.model.WordVector;



public class HotelNameInclusionClassifier2 extends AbstractClassifier implements WeakClassifier{


	private static final long serialVersionUID = -6724315735779212792L;
	private static boolean DEBUG = ClassifierApplication.DEBUG_WEAK_CLASSIFIER_SCORES;
//	private static LevensteinDistance l_algo = new LevensteinDistance();
//	private static Transliterator oTransliteratorForZen2Han = Transliterator.getInstance("Halfwidth-Fullwidth");

//	final EndingPreProcessor preProcessor = new EndingPreProcessor();





    //word2vec辞書読み込み
	static WordVectors vec = null;
	private static Word2VecModel model = Word2VecModel.load("/data/wiki.model");
    private static DataFrame word_dict = model.getVectors();
    private static List<Row> dictionary = word_dict.collectAsList();

    //
    static float Excep_Num = 2;




	private float _threshold = 0f;
	private float _normalizer = 0f;


	@Override
	public void train(DataSet dataSetToTrain) {

		// TODO 自動生成されたメソッド・スタブ

		Collection<Match> matches = dataSetToTrain.getMatches();

		Separation sep = new Separation();
		for(Match match : matches){
			Record[] records = match.getRecords();
			float d = this.distance(records[0].getStringValue("NAME"), records[1].getStringValue("NAME"));
//			float d2 = this.distance(records[1].getStringValue("NAME"), records[0].getStringValue("NAME"));
//			float d = (d1 < d2) ? d1 : d2;

			sep.addValue(d, match.isMatching());

			if(DEBUG){
				System.out.println("Train, " + this.getClass().getSimpleName() + ", " + match.isMatching() + ", " + d + ", " + records[0].getStringValue("NAME") + ", " + records[1].getStringValue("NAME"));
			}
		}


		this._threshold = sep.getThreshold();
		this._normalizer = sep.getNormalizer();
		if(DEBUG){
			System.out.println(this.getClass().getSimpleName() + ", norm=" + this._normalizer + ", threshold=" + this._threshold);
		}
	}

	@Override
	public Match classify(Record record1, Record record2) {
		// TODO 自動生成されたメソッド・スタブ
		String name1 = record1.getStringValue("NAME");
		String name2 = record2.getStringValue("NAME");

		float d = this.distance(name1, name2);
//		float d2 = this.distance(name2, name1);
//		float d = (d1 < d2) ? d1 : d2;
		float s = (d - this._threshold) / this._normalizer;
//		System.out.println("d=" + d + ", s=" + s);

		String id_key = record1.getId() + record2.getId();
//		New_score n = null;
//		if(ClassifierApplication .singletonmap.containsKey(id_key)) {
//			n = ClassifierApplication.singletonmap.get(id_key);
//		}
//		else
//		{
//			n= new New_score();
//			ClassifierApplication.singletonmap.put(id_key, n);
//		}
//
//		n.set_week_score(s);


		return new MatchWithLikelihood(record1, record2, this.getClass(), (s > 0), s);
	}

	@SuppressWarnings("resource")
	public float distance(String name1, String name2){





//      辞書にない場合、随時追加してもよい
//		name1 = name1.replaceAll("株式会社", "");
//		name2 = name2.replaceAll("株式会社", "");





		//記号除去
		name1 =
				name1.replaceAll("["
				+
				Pattern.quote( "[【】（>）！――''］.・*＆+…<＼│％▼［♪◆!☆(｜)●/◇『』※＞＜★%◎：×「」≫≪＊／■:,《》〜＋「」／]1234567890##＃＃１２３４５６７８９０" )
				+
				"]",
				"" );
		name2 =
				name2.replaceAll("["
				+
				Pattern.quote( "[【】（>）！――''］.・*＆+…<＼│％▼［♪◆!☆(｜)●/◇『』※＞＜★%◎：×「」≫≪＊／■:,《》〜＋「」／]1234567890##＃＃１２３４５６７８９０" )
				+
				"]",
				"" );

		float dInclusionOfNameNormalizedT2M = 0.0f;
        float dInclusionOfNameNormalizedM2T = 0.0f;

//		System.out.println(name1 + "<=>" + name2 + ", ");
		if (name1.length() == 0 || name2.length() == 0) {
			// カテゴリがない場合は一致率測定を実施しない。
			return 0f;
		}







			String sudachiPath = "C:\\pleiades2\\workspace\\sudachi\\sudachi\\sudachi_data";
	        String strSudachiSettings;
	        Dictionary dict = null;
			try {
				strSudachiSettings = Files.lines( Paths.get( "C:\\pleiades2\\workspace\\sudachi\\sudachi\\sudachi_data\\sudachi.json" ) ).collect( Collectors.joining());
				// sudachiの設定をきめているfile
				dict = new DictionaryFactory().create(sudachiPath, strSudachiSettings);

			} catch (IOException e) {
				// TODO 自動生成された catch ブロック

				e.printStackTrace();

			}
//	        logger.debug( "strSudachiSettings = " + strSudachiSettings );




	        Tokenizer onlyJapaneseAnalyzer = dict.create();
	        //マルチスレッドの中で使わなければ一つでよい



	        name1 = name1.toLowerCase();
	        name2 = name2.toLowerCase();

	        List<Morpheme> token1 = onlyJapaneseAnalyzer.tokenize(Tokenizer.SplitMode.A ,name1);
	        List<Morpheme> token2 = onlyJapaneseAnalyzer.tokenize(Tokenizer.SplitMode.A, name2);

	        dInclusionOfNameNormalizedT2M = calculateInclusionRatio( token1, name2);
	        dInclusionOfNameNormalizedM2T = calculateInclusionRatio(token2, name1);











//		TokenizerFactory  oTargetJapaneseAnalyzer = new DefaultTokenizerFactory();
//		TokenizerFactory  oMasterJapaneseAnalyzer = new DefaultTokenizerFactory();
//		oTargetJapaneseAnalyzer.setTokenPreProcessor(new TokenPreProcess()
//		{
//			public String preProcess( String name1)
//			{
//				name1 = name1.toLowerCase();
//				String base = preProcessor.preProcess(name1);
//				return base;
//			}
//		});
//		oMasterJapaneseAnalyzer.setTokenPreProcessor(new TokenPreProcess()
//		{
//			public String preProcess( String name2)
//			{
//				name2 = name2.toLowerCase();
//				String base2 = preProcessor.preProcess(name2);
//				return base2;
//			}
//
//		});
//      dInclusionOfNameNormalizedT2M = calculateInclusionRatio( oTargetJapaneseAnalyzer, name2 );
//      dInclusionOfNameNormalizedM2T = calculateInclusionRatio( oMasterJapaneseAnalyzer, name1 );
//
//
//
//
//
////        JapaneseAnalyzer oTargetJapaneseAnalyzer =
////            new JapaneseAnalyzer( null, JapaneseTokenizer.Mode.NORMAL, JapaneseAnalyzer.getDefaultStopSet(), JapaneseAnalyzer.getDefaultStopTags() );
////        JapaneseAnalyzer oMasterJapaneseAnalyzer =
////            new JapaneseAnalyzer( null, JapaneseTokenizer.Mode.NORMAL, JapaneseAnalyzer.getDefaultStopSet(), JapaneseAnalyzer.getDefaultStopTags() );
////        TokenStream oTokenStreamOfName1 = oTargetJapaneseAnalyzer.tokenStream( "", new StringReader( name1 ) );
////        TokenStream oTokenStreamOfName2 = oMasterJapaneseAnalyzer.tokenStream( "", new StringReader( name2 ) );
////
////
////
////        dInclusionOfNameNormalizedT2M = calculateInclusionRatio( oTokenStreamOfName1, name2 );
////        dInclusionOfNameNormalizedM2T = calculateInclusionRatio( oTokenStreamOfName2, name1 );
////
        if (dInclusionOfNameNormalizedT2M >= dInclusionOfNameNormalizedM2T){
        	return dInclusionOfNameNormalizedT2M;
        }
        else {
        	return dInclusionOfNameNormalizedM2T;
        }
	}

	public static float calculateInclusionRatio(List<Morpheme> tokens, String name)
	{
		float dInclusionCount = 0.0f;
		for(Morpheme morpheme :tokens) {
			if(-1 < name.indexOf(morpheme.surface()))
			{
				dInclusionCount+= makeOmomi(morpheme.surface());
			}
		}

		return dInclusionCount/(float) tokens.size();

	}




//
//	public static float calculateInclusionRatio( TokenizerFactory oTargetTokenStream, String strReference )
//    {
//        HashSet<String> oTargetHashSet    = new HashSet<String>();
////        ArrayList<String> oTargetHashSet    = new ArrayList<String>();
//
////        try
////        {
////            CharTermAttribute oTargetCharTermAttribute = oTargetTokenStream.addAttribute( CharTermAttribute.class );
////            oTargetTokenStream.reset();
////            while ( oTargetTokenStream.incrementToken() )
////            {
////                oTargetHashSet.add( oTargetCharTermAttribute.toString() );
////            }
////
//            float dInclusionCount = 0.0f;
////            for ( String strTmp : oTargetHashSet )
////            {
////                if ( -1 < strReference.indexOf( strTmp ) )
////                {
////                    dInclusionCount++;
////                }
////            }
//            return dInclusionCount / (float)oTargetHashSet.size();
////        }
////        catch (Exception ex)
////        {
////            return 0.0f;
////        }
//    }
	public static float makeOmomi(String _word_) {

		try {
			WordVectors vec = WordVector.getWordVectors();
			double[] vector = vec.getWordVector(_word_);
			double[] determined_vector = {-0.738346878,0.442009503,3.240274954,-0.42393345,2.556297139,0.928465907,-2.482083304,-4.706446229,3.395536477,-3.477390284,2.673662313,-1.087584217,-7.101648449,-4.57923012,-0.251181862,4.189618895,-1.111209933,1.746571919,-3.035865599,-2.068907019,3.395095826,-0.626106886,3.648003927,-4.563727906,-2.69490985,-0.172650972,-0.841743792,2.318056659,0.560476154,-2.700815443,2.640475118,-2.413307274,-1.260440407,-2.85048922,-1.886473506,4.41491019,3.044953307,1.727466619,-3.146617481,-2.186045412,3.67126735,-3.802552444,-2.784906637,-0.724100476,-1.839810248,-0.490904071,-1.535249269,1.533067497,-0.388883682,-1.58072366,2.526488116,-1.175144697,-1.781324174,-2.565752792,-3.050033224,-2.856496407,-4.252336784,1.617282681,-0.861524919,2.102913069,0.756652939,-1.383223618,-0.626763734,-0.20554413,2.993275391,1.201198745,3.036123683,2.296338283,0.549100996,-0.57666704,-2.381594522,-2.787222802,-0.659049165,3.865510944,-1.306409479,2.798513837,0.882007936,-3.222497315,0.385644932,2.448842234,0.121628342,0.845623439,-2.460121227,1.709603615,-1.128632333,0.591011945,0.708749847,-0.268061999,2.139378683,0.10891646,-0.090229639,-0.612570166,0.783807178,0.732165258,-1.233683751,-2.253010379,-0.901797635,-2.083870283,-5.351092903,2.203141191,2.317121001,0.974916262,-1.149391734,-3.750180961,1.883671567,-4.123093542,1.002872116,-1.682846091,-0.163748637,-3.351070765,-3.467095892,0.211446289,-0.899554373,0.899075143,-3.782335882,1.89379804,-0.263255666,-0.817266903,1.370447264,3.24505654,-0.34922156,2.48207232,-0.710407888,1.671651515,0.017325609,-5.510479717,3.966148425,0.725697654,-0.035066502,1.704881023,-2.193841177,1.178347424,1.191274871,-0.036915341,5.877206523,-5.822176746,2.421616288,-0.643719167,2.76029032,-0.558076045,-2.896634381,-0.334858429,-1.814745362,-3.11465907,-3.115403483,-2.927585739,-2.289043553,0.966444388,-0.513613232,-1.182506972,-1.191349124,0.162404214,-1.929531581,-1.331561959,-1.247758371,3.63774019,-6.287638338,3.27912137,-0.811805772,-0.792907952,-3.05584557,-1.796858002,-3.107939991,-0.119443965,-1.958763902,2.322851259,2.966689623,3.273024146,-1.171303758,3.922359307,1.188990318,0.355374259,-1.294665442,2.775088852,-1.355051416,0.538928216,-4.396700871,-1.833889697,-0.615641293,1.210472497,-1.457600836,1.174436103,-4.460449801,1.515092516,-1.619086039,1.620683213,1.705857009,0.751920554,0.575678042,2.372627809,-0.805219371,3.35034541,-1.543574291,-2.461188227,1.392189529,3.660677686,2.156529641,-1.080320885,-1.168809376,0.239270336};
			float dist = cosineSimilarity(vector, determined_vector);
			if(dist > 0.26)//厳
//			if(dist > 0.145)
			{
//				return 2*dist;
				return (1-dist)/ HotelNameInclusionClassifier2.Excep_Num;
//				return 4*dist;

			}
			else
			{
				float det= 2.0f;
				return det;
//				return ClassifierApplication.Excep_Num;
//				return 1;
			}

//			float var_x = 1- dist;
//			if(var_x< 0.9)
//			{
//				return (float) ((-1)/(2*((var_x/3)+0.4)) +1.5);
//
//			}
//			else
//			{
//				return 1;
//			}


		}catch(Exception e) {
//          辞書に載ってないwordを表示
//			System.out.println(_word_);
			return 4;

//			return ClassifierApplication.Excep_Num;

		}


	}

	public static float cosineSimilarity(double[] vectorA, double[] vectorB) {
	    float dotProduct = 0.0f;
	    float normA = 0.0f;
	    float normB = 0.0f;
	    for (int i = 0; i < vectorB.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }
	    return (float)(dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
	}


	@Override
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append(this.getClass().getSimpleName());
		sb.append("{t=");
		sb.append(this._threshold);
		sb.append(", n=");
		sb.append(this._normalizer);
		sb.append("}");
		return sb.toString();
	}


}
