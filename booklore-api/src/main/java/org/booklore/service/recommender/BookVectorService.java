package org.booklore.service.recommender;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ko.KoreanAnalyzer;
import org.apache.lucene.analysis.ko.KoreanTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.MorphosyntacticAnalysisAttribute;
import org.booklore.model.entity.AuthorEntity;
import org.booklore.model.entity.BookEntity;
import org.booklore.model.entity.BookMetadataEntity;
import org.booklore.model.entity.CategoryEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class BookVectorService {

    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final int VECTOR_DIMENSION = 128;
    private static final Analyzer KOREAN_ANALYZER =
            new KoreanAnalyzer(null, KoreanTokenizer.DecompoundMode.NONE, null, false);

    public double[] generateEmbedding(BookEntity book) {
        if (book.getMetadata() == null) {
            return new double[VECTOR_DIMENSION];
        }

        BookMetadataEntity metadata = book.getMetadata();
        Map<String, Double> features = new HashMap<>();

        if (metadata.getTitle() != null) {
            addTextFeatures(features, "title", metadata.getTitle(), 3.0);
        }

        if (metadata.getAuthors() != null) {
            metadata.getAuthors().stream()
                    .map(AuthorEntity::getName)
                    .filter(Objects::nonNull)
                    .forEach(author -> features.put("author_" + author.toLowerCase(), 5.0));
        }

        if (metadata.getCategories() != null) {
            metadata.getCategories().stream()
                    .map(CategoryEntity::getName)
                    .filter(Objects::nonNull)
                    .forEach(cat -> features.put("category_" + cat.toLowerCase(), 4.0));
        }

        if (metadata.getSeriesName() != null) {
            features.put("series_" + metadata.getSeriesName().toLowerCase(), 6.0);
        }

        if (metadata.getPublisher() != null) {
            features.put("publisher_" + metadata.getPublisher().toLowerCase(), 2.0);
        }

        if (metadata.getDescription() != null) {
            addTextFeatures(features, "desc", metadata.getDescription(), 1.0);
        }

        return featuresToVector(features);
    }

    private boolean isNoiseToken(String token, String pos) {
        // "나"/"내" + NP(대명사): 웹소설에서 흔한 1인칭 노이즈
        if (("나".equals(token) || "내".equals(token)) && pos.startsWith("NP")) return true;
        // "하"/"되" + VV(동사): 의미 없는 일반 동사
        if (("하".equals(token) || "되".equals(token)) && pos.startsWith("VV")) return true;
        // "하" + VX(보조동사): "가야 했다"의 보조동사
        if ("하".equals(token) && pos.startsWith("VX")) return true;
        // "이" + VCP(긍정지정사): "미륵이니라"의 지정사
        if ("이".equals(token) && pos.startsWith("VCP")) return true;
        return false;
    }

    private void addTextFeatures(Map<String, Double> features, String prefix, String text, double weight) {
        try (TokenStream stream = KOREAN_ANALYZER.tokenStream("content", text.toLowerCase())) {
            CharTermAttribute charTermAttr = stream.addAttribute(CharTermAttribute.class);
            MorphosyntacticAnalysisAttribute morphAttr =
                    stream.addAttribute(MorphosyntacticAnalysisAttribute.class);
            stream.reset();

            int count = 0;
            while (stream.incrementToken() && count < 20) {
                String token = charTermAttr.toString();
                String pos = morphAttr.getMorphosyntacticAnalysis();

                if (!isNoiseToken(token, pos)) {
                    features.merge(prefix + "_" + token, weight, Double::sum);
                    count++;
                }
            }
        } catch (IOException e) {
            log.warn("Failed to tokenize text for features: {}", text, e);
        }
    }

    private double[] featuresToVector(Map<String, Double> features) {
        double[] vector = new double[VECTOR_DIMENSION];

        for (Map.Entry<String, Double> entry : features.entrySet()) {
            int hash = Math.abs(entry.getKey().hashCode());
            int index = hash % VECTOR_DIMENSION;
            vector[index] += entry.getValue();
        }

        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);

        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }

        return vector;
    }

    public String serializeVector(double[] vector) {
        try {
            return objectMapper.writeValueAsString(vector);
        } catch (JacksonException e) {
            log.error("Error serializing vector", e);
            return null;
        }
    }

    public double[] deserializeVector(String vectorJson) {
        if (vectorJson == null || vectorJson.isEmpty()) {
            return null;
        }
        try {
            return objectMapper.readValue(vectorJson, double[].class);
        } catch (JacksonException e) {
            log.error("Error deserializing vector", e);
            return null;
        }
    }

    public double cosineSimilarity(double[] v1, double[] v2) {
        if (v1 == null || v2 == null || v1.length != v2.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
        }

        return dotProduct;
    }

    public List<ScoredBook> findTopKSimilar(double[] targetVector, List<ScoredBook> candidates, int k) {
        if (targetVector == null) {
            return Collections.emptyList();
        }

        return candidates.stream()
                .sorted(Comparator.comparingDouble(ScoredBook::getScore).reversed())
                .limit(k)
                .collect(Collectors.toList());
    }

    @Getter
    public static class ScoredBook {
        private final Long bookId;
        private final double score;

        public ScoredBook(Long bookId, double score) {
            this.bookId = bookId;
            this.score = score;
        }

    }
}

