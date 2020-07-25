package JavaExtractor.Common;

import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<Node> leaves;
    private final String name;

    private final String content;

    public MethodContent(ArrayList<Node> leaves, String name, String content) {
        this.leaves = leaves;
        this.name = name;
        this.content = content;
    }

    public ArrayList<Node> getLeaves() {
        return leaves;
    }

    public String getName() {
        return name;
    }

    public String getContent() {
        return content;
    }
}
