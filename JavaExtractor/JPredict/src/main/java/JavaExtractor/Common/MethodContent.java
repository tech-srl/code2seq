package JavaExtractor.Common;

import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<Node> leaves;
    private final String name;

    public MethodContent(ArrayList<Node> leaves, String name) {
        this.leaves = leaves;
        this.name = name;
    }

    public ArrayList<Node> getLeaves() {
        return leaves;
    }

    public String getName() {
        return name;
    }
}
