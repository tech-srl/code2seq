package JavaExtractor.FeaturesEntities;

public class ProgramRelation {
    private final Property m_Source;
    private final Property m_Target;
    private final String m_Path;

    public ProgramRelation(Property sourceName, Property targetName, String path) {
        m_Source = sourceName;
        m_Target = targetName;
        m_Path = path;
    }

    public String toString() {
        return String.format("%s,%s,%s", m_Source.getName(), m_Path,
                m_Target.getName());
    }
}
