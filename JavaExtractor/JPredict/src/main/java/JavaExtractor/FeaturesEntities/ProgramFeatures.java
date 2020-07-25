package JavaExtractor.FeaturesEntities;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.Collectors;

public class ProgramFeatures {
    String name;

    transient ArrayList<ProgramRelation> features = new ArrayList<>();
    String textContent;

    String filePath;

    public ProgramFeatures(String name, Path filePath, String textContent) {

        this.name = name;
        this.filePath = filePath.toAbsolutePath().toString();
        this.textContent = textContent;
    }

    @SuppressWarnings("StringBufferReplaceableByString")
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append(" ");
        stringBuilder.append(features.stream().map(ProgramRelation::toString).collect(Collectors.joining(" ")));

        return stringBuilder.toString();
    }

    public void addFeature(Property source, String path, Property target) {
        ProgramRelation newRelation = new ProgramRelation(source, target, path);
        features.add(newRelation);
    }

    public boolean isEmpty() {
        return features.isEmpty();
    }
}
