package jianzhi️offer;

/**
 * 剑指offer上的面试题
 */
public class JianZhiOfferSolution {

    public static void main(String[] args) {
        int[] nums = {2, 3, 5, 4, 3, 2, 6, 7};
        int[][] matrix = {
                {1, 2, 8, 9},
                {2, 4, 9, 12},
                {4, 7, 10, 13},
                {6, 8, 11, 15}
        };
        int duplication = new JianZhiOfferSolution().getDuplication(nums, 8);
        boolean found = new JianZhiOfferSolution().Find(matrix, 4, 4, 7);
        System.out.println(found);

    }

    // ================================================================================
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false

    /**
     * 找出数组中重复的数字
     */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        if (numbers == null || numbers.length < 0) {
            return false;
        }
        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] < 0 || numbers[i] > length - 1) {
                return false;
            }
        }
        for (int i = 0; i < length; i++) {
            while (numbers[i] != i) {
                if (numbers[i] == numbers[numbers[i]]) {
                    duplication[0] = numbers[i];
                    return true;
                }
                // swap
                int temp = numbers[i];
                numbers[i] = numbers[temp];
                numbers[temp] = temp;
            }
        }
        return false;
    }

    // ================================================================================

    /**
     * 不修改数组找出重复的数字
     * 缺点：不能找到所有重复的数字
     *
     * @param nums
     * @param length
     * @return
     */
    int getDuplication(int[] nums, int length) {
        if (nums == null || length <= 0) {
            return -1;
        }
        int start = 1;
        int end = length - 1;
        while (end >= start) {
            int middle = ((end - start) >> 1) + start;
            int count = countRange(nums, length, start, middle);
            if (end == start) {
                if (count > 1) {
                    return start;
                } else {
                    break;
                }
            }
            if (count > (middle - start + 1)) {
                end = middle;
            } else {
                start = middle + 1;
            }
        }
        return -1;
    }

    public int countRange(int[] nums, int length, int start, int end) {
        if (nums == null) {
            return 0;
        }
        int count = 0;
        for (int i = 0; i < length; i++) {
            if (nums[i] >= start && nums[i] <= end) {
                count++;
            }
        }
        return count;
    }

    // ================================================================================

    /**
     * 在一个二维数组中（每个一维数组的长度相同），
     * 每一行都按照从左到右递增的顺序排序，
     * 每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数，
     * 判断数组中是否含有该整数。
     * @param matrix
     * @param rows
     * @param cols
     * @param num
     * @return
     */
    boolean Find(int[][] matrix, int rows, int cols, int num) {
        boolean found = false;
        if (matrix != null && rows > 0 && cols > 0) {
            int row = 0;
            int col = cols-1;
            while (row < rows && col >= 0) {
                if (matrix[row][col] == num) {
                    found = true;
                    break;
                } else if (matrix[row][col] > num) {
                    col--;
                } else {
                    row++;
                }

            }
        }
        return found;
    }


    // ================================================================================
}