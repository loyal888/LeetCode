package jianzhi️offer;

/**
 * 剑指offer上的面试题
 */
public class JianZhiOfferSolution {

    public static void main(String[] args) {
        int[] nums = {2, 3, 5, 4, 3, 2, 6, 7};
        int duplication = new JianZhiOfferSolution().getDuplication(nums, 8);
        System.out.println(duplication);

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
}