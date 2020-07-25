package JavaExtractor.FeaturesEntities;

public class ProgramRelation {
    Property source;
    Property target;
    String path;

    public ProgramRelation(Property sourceName, Property targetName, String path) {
        source = sourceName;
        target = targetName;
        this.path = path;
    }

    public String toString() {
        return String.format("%s,%s,%s", source.getName(), path,
                target.getName());
    }
}
