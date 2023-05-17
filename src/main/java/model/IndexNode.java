package model;

import java.util.ArrayList;
import java.util.List;

public class IndexNode {
    private final String name;
    private final double time;
    private final List<IndexNode> children;

    public IndexNode(String name, double time) {
        this.name = name;
        this.time = time;
        this.children = new ArrayList<>();
    }

    public void addChildren(IndexNode node) {
        this.children.add(node);
    }

    public double getTime() {
        return time;
    }
    public String getName() {return name;}

    public boolean isLeaf() {
        return children.isEmpty();
    }

    public List<IndexNode> getChildren() {
        return children;
    }

    @Override
    public String toString() {
        return name;
    }
}
