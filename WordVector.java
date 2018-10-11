package jp.jobdirect.dbmatching.model;

import java.io.File;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class WordVector {
	private static WordVectors vector = null;
	public static WordVectors getWordVectors()
	{
		if (vector != null) {
			return vector;
		}

		vector = makeVector();
		return vector;

	}
	private static  WordVectors makeVector()
	{
		try
		{

//			File is = new File("C:/Users/shun_umetsu/Desktop/test.model");
//			WordVectors vec = WordVectorSerializer.loadTxtVectors(is);


			 File gModel = new File ( "C:/Users/shun_umetsu/Desktop/umetsu.model" );

			 Word2Vec vec = WordVectorSerializer.readWord2VecModel(gModel);
			return vec;
		}
		catch(Exception e)
		{
			System.out.println("モデルファイルの読み込みに失敗しました");
			return null;
		}
	}

}
