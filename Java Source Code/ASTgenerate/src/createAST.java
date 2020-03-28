import org.apache.commons.io.FileUtils;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.*;

import java.io.File;
import java.util.Map;
import java.util.*;
import java.io.*;

public class createAST {
    public static void main(String[] args) {
        ASTParser parser = ASTParser.newParser(AST.JLS11);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);

        parser.setCompilerOptions(null);
        parser.setResolveBindings(true);

        Map<String, String> compilerOptions = JavaCore.getOptions();
        compilerOptions.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_1_8); 
        compilerOptions.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_1_8);
        compilerOptions.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_1_8);
        parser.setCompilerOptions(compilerOptions); 

        String src = null;
        String input_file = args[0];
        String output_file = args[1];
        try {
            src = FileUtils.readFileToString(new File(input_file),"UTF-8");  
        } catch (Exception e) {
            e.printStackTrace();
        }
        parser.setSource(src.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);  
        new createAST().print(cu, cu,"", output_file);
    }
    private void print(final CompilationUnit cu, ASTNode node, String blanks, String output_file) {
        List properties = node.structuralPropertiesForType();
        try {
        	
            for (Iterator iterator = properties.iterator(); iterator.hasNext();) {
                Object descriptor = iterator.next();
                if (descriptor instanceof SimplePropertyDescriptor) {
                    SimplePropertyDescriptor simple = (SimplePropertyDescriptor) descriptor;
                    Object value = node.getStructuralProperty(simple);
                    String[] names = simple.getNodeClass().toString().split("\\.");
                    FileWriter out = new FileWriter (output_file, true);
                    out.write(blanks + names[names.length - 1] + ";" +  simple.getId() + ";" + value.toString()+ ";" + cu.getLineNumber(node.getStartPosition()) + "\r\n");
                    out.close();
                } else if (descriptor instanceof ChildPropertyDescriptor) {
                    ChildPropertyDescriptor child = (ChildPropertyDescriptor) descriptor;
                    String[] namesc = child.getNodeClass().toString().split("\\.");
                    ASTNode childNode = (ASTNode) node.getStructuralProperty(child);
                    if (childNode != null) {
                    	FileWriter out_1 = new FileWriter (output_file, true);
                    	out_1.write(blanks + "" +  namesc[namesc.length - 1] + ";" + child.getId() + ":{" + "\r\n");
                    	out_1.close();
                        print(cu, childNode, blanks + "    ", output_file);
                        FileWriter out_2 = new FileWriter (output_file, true);
                        out_2.write(blanks + "}" + "\r\n");
                        out_2.close();
                    }
                } else {
                    ChildListPropertyDescriptor list = (ChildListPropertyDescriptor) descriptor;
                    List check = (List) node.getStructuralProperty(list);
                    String[] namescl = list.getNodeClass().toString().split("\\.");
                    if (check.size() != 0 ) {
                    	FileWriter out_1 = new FileWriter (output_file, true);
                    	out_1.write(blanks + "" + namescl[namescl.length - 1] + ";" + list.getId() + ":{" + "\r\n");
                    	out_1.close();
                    	print_2(cu, (List) node.getStructuralProperty(list), blanks + "    ", output_file);
                    	FileWriter out_2 = new FileWriter (output_file, true);
                    	out_2.write(blanks + "}" + "\r\n");
                    	out_2.close();
                    }
                }
            }
        } catch (IOException ex) {
        	System.out.println(ex);
        }

    }
    private void print_2(final CompilationUnit cu, List nodes, String blanks, String output_file) {
        for (Iterator iterator = nodes.iterator(); iterator.hasNext();) {
            print(cu, (ASTNode) iterator.next(), blanks,  output_file);
        }
    }
    
}