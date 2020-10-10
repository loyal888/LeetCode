package NEW_JOURNEY_IN_WORK.array;

public class LC_977 {
    /**
     * 双指针,时间复杂度O(N),空间复杂度O（N）
     * @param A
     * @return
     */
    public static  int[] sortedSquares(int[] A) {
        int[] newA = new int[A.length];
        int left = 0;
        int right = A.length-1;
        int index = A.length-1;
        while(left<=right){
            int leftDouble = A[left] * A[left];
            int rightDouble = A[right] * A[right];
            if(leftDouble >  rightDouble){
                newA[index] = leftDouble;
                left++;
            }else{
                newA[index] = rightDouble;
                right--;
            }
            index--;
        }
        return newA;
    }

    public static void main(String[] args) {
        sortedSquares(new int[]{-1,0,1,2});
    }
}
