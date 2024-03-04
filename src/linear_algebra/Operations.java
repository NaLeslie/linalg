package linear_algebra;

/**
 * Contains a list of matrix and vector operations as static methods.
 * Matrices are addressed as <code>array[row#][col#]</code> starting at <code>[0][0]</code>.
 * @author Nathaniel
 */
public class Operations {
    /**
     * Inverts matrix <code>A</code> using Gaussian elimination
     * @param A The matrix to be inverted
     * @return The inverse of <code>A</code> (<code>multiply(invert(A),A)</code> will return the identity matrix)
     * @throws SingularMatrixException if a row contains values less than {@link #CUTOFF CUTOFF} during the elimination, this error will be thrown
     */
    public static double[][] invert(double[][] A) throws SingularMatrixException{
        int rows = A.length;
        int cols = A[0].length;
        
        double[][] inverse = new double[rows][rows];
        
        for(int i = 0; i < rows; i++){
            inverse[i][i] = 1;
        }
        
        if(rows == cols){
            for(int i = 0; i < rows; i++){
                //ensure leading value is nonzero
                if(Math.abs(A[i][i]) < CUTOFF){//leading value in the column is zero
                    boolean found = false;
                    for(int j = i+1; j < rows; j++){
                        if(Math.abs(A[j][i]) > CUTOFF){
                            rowOp_swap(A, i, j);
                            rowOp_swap(inverse, i, j);
                            found = true;
                            break;
                        }
                    }
                    if(!found){
                        throw new SingularMatrixException("Row " + i + " not linearly independant.");
                    }
                }
                
                //divide row by leading value
                rowOp_multiply(inverse, 1 / A[i][i], i);
                rowOp_multiply(A, 1 / A[i][i], i);
                
                
                //subtract by leading value
                for(int j = i+1; j < rows; j++){
                    double factor = -A[j][i];
                    rowOp_add(inverse, factor, i, j);
                    rowOp_add(A, factor, i, j);
                }
            }
            
            //move back up the matrix subtracing by the leading value on the bottom of the pivot
            for(int i = rows - 1; i >= 1; i--){
                for(int j = i - 1; j >= 0; j--){
                    double factor = -A[j][i];
                    rowOp_add(inverse, factor, i, j);
                    rowOp_add(A, factor, i, j);
                }
            }
        }
        else{
            throw new SingularMatrixException("Matrix is not square.");
        }
        
        return inverse;
    }
    
    /**
     * Adds <code>s*row_2</code> to <code>row_1</code> in matrix <code>A</code>. 
     * These changes are made directly to <code>A</code> as well as returned
     * @param A The matrix where the row operations will occur.
     * @param s A scalar factor to multiply with <code>row_2</code> as it is added to <code>row_1</code>.
     * @param row_1 the index of the row to be added to.
     * @param row_2 the index of the row whose values will be added to <code>row_1</code>.
     * @return The matrix <code>A</code> after the addition has taken place.
     */
    private static double[][] rowOp_add(double[][] A, double s, int row_1, int row_2){
        int rows = A.length;
        int cols = A[0].length;
        if(row_1 < rows && row_2 < rows && row_1 != row_2){
            for(int j = 0; j < cols; j ++){
                A[row_2][j] += s*A[row_1][j];
            }
        }
        return A;
    }
    
    /**
     * Swaps the position of rows <code>row_1</code> and <code>row_2</code> in matrix <code>B</code>.
     * These changes are made directly to <code>B</code> as well as returned.
     * @param B The matrix in which the rows are to be swapped.
     * @param row1 the first row.
     * @param row2 the second row.
     * @return The matrix <code>B</code> with swapped rows.
     */
    private static double[][] rowOp_swap(double[][] B, int row1, int row2){
        int rows = B.length;
        int cols = B[0].length;
        if(row1 < rows && row2 < rows){
            double temp;
            for(int i = 0; i < cols; i++){
                temp = B[row1][i];
                B[row1][i] = B[row2][i];
                B[row2][i] = temp;
            }
        }
        return B;
    }
    
    /**
     * Multiplies the row <code>row</code> in matrix <code>B</code> with the scalar value <code>s</code>.
     * These changes are made directly to B as well as returned.
     * @param B The matrix in which the row operation is to take place
     * @param s The scalar value that will be multiplied to row <code>row</code>.
     * @param row The row to be affected.
     * @return The matrix B with the multiplied row. 
     */
    private static double[][] rowOp_multiply(double[][] B, double s, int row){
        int rows = B.length;
        int cols = B[0].length;
        if(rows > row){
            for(int j = 0; j < cols; j++){
                B[row][j] = B[row][j]*s;
            }
        }
        return B;
    }
    
    /**
     * Inverts hermitian matrix <code>A</code> using Cholesky decomposition.
     * @param A The hermitian matrix to be inverted
     * @return The inverse of <code>A</code> (<code>multiply(invert(A),A)</code> will return the identity matrix) 
     */
    public static double[][] invertHermitian(double[][] A){
        //Cholesky decomposition following the Cholesky–Banachiewicz algorithm
        int sz = A.length;
        double[][] L = choleskyHermitian(A);
        
        //M = L-1 = (m1,m2,...,mn)
        //LM = I = (e1,e2,...en)
        //Lmi = ei
        double[][] M = new double[sz][sz];
        for(int i = 0; i < sz; i++){
            double[] ei = new double[sz];
            ei[i] = 1;
            
            for(int j = 0; j < sz; j++){
                double sum = 0.0;
                for(int k = 0; k < j; k++){
                    sum += L[j][k]*M[k][i];
                }
                M[j][i] = (ei[j] - sum)/L[j][j];
            }
            
        }
        
        //A=LLT
        //A-1=(LLT)-1
        //A-1=(LT)-1(L-1)
        //A-1=(L-1)T(L-1)
        //A-1=MTM
        double[][] MT = transpose(M);
        
        return multiply(MT, M);
    }
    
    /**
     * Returns an identity matrix of the specified size.
     * @param size The size of the identity matrix to be created.
     * @return An identity matrix of size <code>size</code>.
     */
    public static double[][] identity(int size){
        double[][] ident = new double[size][size];
        for(int i = 0; i < size; i++){
            ident[i][i] = 1;
        }
        return ident;
    }
    
    /**
     * Transposes the matrix <code>A</code> (swaps the columns and rows).
     * @param A The matrix to be transposed.
     * @return The transposed matrix.
     */
    public static double[][] transpose(double[][] A){
        int sz = A.length;
        double[][] transposed = new double[sz][sz];
        for(int i = 0; i < sz; i++){
            for(int j = 0; j < sz; j ++){
                transposed[i][j] = A[j][i];
            }
        }
        return transposed;
    }
    
    /**
     * Multiplies matrices <code>A*B</code>.
     * @param A The left matrix in the multiplication.
     * @param B The right matrix in the multiplication.
     * @return The product matrix of the multiplication.
     */
    public static double[][] multiply(double[][] A, double[][] B){
        int rows = A.length;
        int intermed = A[0].length;
        int cols = B[0].length;
        double[][] output = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                double sum = 0.0;
                for(int k = 0; k < intermed; k++){
                    sum += A[i][k]*B[k][j];
                }
                output[i][j] = sum;
            }
        }
        return output;
    }
    
    /**
     * Multiplies <code>A*v</code>.
     * @param A The left matrix of the multiplication.
     * @param v The right vector of the multiplication.
     * @return The resulting vector of the multiplication of matrix <code>A</code> with vector <code>v</code>.
     */
    public static double[] multiply(double[][] A, double[] v){
        int rows = A.length;
        int cols = A[0].length;
        double[] output = new double[rows];
        for(int i = 0; i < rows; i++){
            double sum = 0.0;
            for(int k = 0; k < cols; k++){
                sum += A[i][k]*v[k];
            }
            output[i] = sum;
        }
        return output;
    }
    
    /**
     * Multiplies <code>s*A</code>.
     * @param A The matrix to be multiplied.
     * @param s The scalar to be multiplied.
     * @return The product of matrix <code>A</code> and scalar <code>s</code>.
     */
    public static double[][] multiply(double[][] A, double s){
        int rows = A.length;
        int cols = A[0].length;
        double[][] output = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[i][j] = A[i][j]*s;
            }
        }
        return output;
    }
    
    /**
     * Multiplies <code>s*v</code>.
     * @param v The vector to be multiplied.
     * @param s The scalar to be multiplied.
     * @return The product of vector <code>v</code> and scalar <code>s</code>.
     */
    public static double[] multiply(double[] v, double s){
        int sz = v.length;
        double[] output = new double[sz];
        for(int i = 0; i < sz; i++){
            output[i] = v[i]*s;
        }
        return output;
    }
    
    /**
     * Computes the inner or 'dot' product of vectors <code>a</code> and <code>b</code>.
     * @param a One of the vectors in the inner product.
     * @param b One of the vectors in the inner product.
     * @return The scalar result of the inner product of <code>a</code> and <code>b</code>.
     */
    public static double innerProduct(double[] a, double[] b){
        double sum = 0.0;
        for(int i = 0; i < a.length; i++){
            sum += a[i]*b[i];
        }
        return sum;
    }
    
    /**
     * Computes the outer product of vectors <code>a</code> and <code>b</code>.
     * @param a One of the vectors in the inner product.
     * @param b One of the vectors in the inner product.
     * @return The matrix result of the outer product of <code>a</code> and <code>b</code>.
     */
    public static double[][] outerProduct(double[] a, double[] b){
        int rows = a.length;
        int cols = b.length;
        double[][] output = new double[rows][cols];
        for(int i = 0; i < rows; i ++){
            for(int j = 0; j < cols; j++){
                output[i][j] = a[i]*b[j];
            }
        }
        return output;
    }
    
    /**
     * Computes the trace of matrix <code>A</code>.
     * @param A The matrix whose diagonal components will be summed.
     * @return the trace of matrix <code>A</code>.
     */
    public static double trace(double[][] A){
        double sum = 0.0;
        for(int i = 0; i < A.length; i++){
            sum += A[i][i];
        }
        return sum;
    }
    
    /**
     * Computes the sum of matrices <code>A</code> and <code>B</code>.
     * @param A One of the matrices in the sum.
     * @param B One of the matrices in the sum.
     * @return The matrix representing the sum <code>A+B</code>.
     */
    public static double[][] add(double[][] A, double[][] B){
        int rows = A.length;
        int cols = A[0].length;
        double[][] output = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[i][j] = A[i][j] + B[i][j];
            }
        }
        return output;
    }
    
    /**
     * Computes the difference <code>A-B</code>.
     * @param A The matrix to be subtracted from.
     * @param B The matrix by which <code>A</code> will be subtracted.
     * @return the matrix representing the difference <code>A-B</code>.
     */
    public static double[][] subtract(double[][] A, double[][] B){
        int rows = A.length;
        int cols = A[0].length;
        double[][] output = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[i][j] = A[i][j] - B[i][j];
            }
        }
        return output;
    }
    
    /**
     * Computes the determinant of matrix <code>A</code>, <code>|A|</code>.
     * This is achieved by multiplying A with its transpose and performing Cholesky decomposition to produce diagonal matrices to quickly compute the determinant.
     * @param A the matrix from which the determinant is to be calculated.
     * @return the determinant of matrix <code>A</code>.
     */
    public static double determinant(double[][] A){
        double[][] AT = transpose(A);
        double[][] ATA = multiply(AT, A);
        double detsq = determinantHermitian(ATA);
        return Math.sqrt(detsq);
    }
    
    /**
     * Computes the determinant of hermitian matrix <code>A</code>, <code>|A|</code>.
     * This is achieved by performing Cholesky decomposition to produce diagonal matrices to quickly compute the determinant.
     * @param A the hermitian matrix from which the determinant is to be calculated.
     * @return the determinant of matrix <code>A</code>.
     */
    public static double determinantHermitian(double[][] A){
        int sz = A.length;
        double[][] L = choleskyHermitian(A);
        double product = 1.0;
        for(int i = 0; i < sz; i++){
            product *= L[i][i];
        }
        return product*product;
    }
    
    /**
     * Computes the lower triangular matrix, <code>L</code> that Cholesky decomposes hermitian matrix <code>A</code> using the Cholesky–Banachiewicz algorithm.
     * <code>A = multiply(L, transpose(L))</code>.
     * @param A The hermitian matrix to be decomposed.
     * @return the lower triangular matrix, <code>L</code> that Cholesky decomposes hermitian matrix <code>A</code>.
     */
    public static double[][] choleskyHermitian(double[][] A){
        //Cholesky decomposition following the Cholesky–Banachiewicz algorithm
        int sz = A.length;
        double[][] L = new double[sz][sz];//lower triangular matrix A = LLT = LTL = AT
        
        for(int i = 0; i < sz; i++){
            for(int j = 0; j <= i; j++){
                
                double sum = 0.0;
                for(int k = 0; k < j; k++){
                    sum += L[i][k] * L[j][k];
                }
                if (i == j){
                    L[i][j] = Math.sqrt(A[i][i] - sum);
                }
                else{
                    L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum));
                }
                
            }
        }
        return L;
    }
    
    /**
     * Computes the square of the L2 norm of the vector <code>v</code>. This is achieved by adding the square of each vector component.
     * @param v the vector whose squared L2 norm is to be computed.
     * @return The square of the L2 norm of <code>v</code>.
     */
    public static double l2norm_square(double[] v){
        double sum = 0.0;
        for(int i = 0; i < v.length; i++){
            sum += v[i]*v[i];
        }
        return sum;
    }
    
    /**
     * Prints the contents of matrix <code>A</code> to <code>System.out</code>.
     * @param A The matrix to be printed.
     */
    public static void printMatrix(double[][] A){
        int rows = A.length;
        int cols = A[0].length;
        
        for(int i = 0; i < rows; i++){
            
            for(int j = 0; j < cols; j++){
                System.out.print(A[i][j] + "\t");
            }
            System.out.println("");
        }
    }
    
    /**
     * The value below which a matrix value will be considered to be zero.
     */
    public static final double CUTOFF = 1E-25;
}
