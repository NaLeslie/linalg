package linear_algebra;

/**
 * Custom Exception for handling singular matrices.
 * @author Nathaniel
 */
public class SingularMatrixException extends Exception{
    public SingularMatrixException(String message){
        super(message);
    }
    
    public SingularMatrixException(){
        super("Matrix is singular");
    }
}