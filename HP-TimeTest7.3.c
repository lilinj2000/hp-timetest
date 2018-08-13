// usage:  [-m,  --method "time"|"cycles"(default="time")]
//         [-t,  --threshold #(default=10 usecs|10000 cycles)]
//         [-l,  --loopcount #(default=5000000000 (time)|5000000000 (cycles))]
//         [-f,  --format "csv"|"xml"|"freeform"(default=freeform)]
//         [-o,  --option "date" "smi_count" "power_hog" "overhead"]
//         [-p,  --priority ["FIFO"|"RR"|"OTHER"(default policy="FIFO")][,#(default priority=sched_get_priority_max(=99 for FIFO,RR))][,#(default nice=-20)]
//         [-V,  --Version]
//         [-v#, --verbose[=#(default=1)] [-b, --brief]
//         [-e,  --explain] [-? -h, --help]

/* Additional things to look into implementing:

nanosecond timer; I think it requires 2.6.26 or later
    rv = clock_gettime( CLOCK_REALTIME, &(TimeSpec[0]) );
    printf("clock_gettime() returned a time of %ld.%9.9ld\n", TimeSpec[0].tv_sec, TimeSpec[0].tv_nsec );

read the value of /proc/sys/kernel/vsyscall64
    0: Provides the most accurate time intervals at μs (microsecond) resolution, but also produces the highest call overhead, as it uses a regular system call 
    1: Slightly less accurate, although still at μs resolution, with a lower call overhead 
    2: The least accurate, with time intervals at the ms (millisecond) level, but offers the lowest call overhead 

Implement ISOLCPUS.  This must be built into the kernel.
There's also a setting in /etc/sysconfig/irqbalance for this; set:
    FOLLOW_ISOLCPUS=yes
...and there's a boot-time parameter to specify which CPUS to isolate.

Consider looking through RedHat's MRG realtime tuning guide:
    http://docs.redhat.com/docs/en-US/Red_Hat_Enterprise_MRG/1.3/html/Realtime_Tuning_Guide/

Consider adding an option to specify what scheduling priority to use.  E.g., a number; or "realtime+" might be 98 while "realtime-" would be 90.
The document at http://kernel.org/doc/ols/2009/ols2009-pages-79-86.pdf is Dominic Duval's (of Red Hat, Inc.)
paper "From Fast to Predictably Fast" which was given at the Linux Symposium on 13-17 July 2009 in Montreal, Canada
In it he offers this table of typical Real Time Priorities:
99	Watchdog and migration threads
90-98	High priority realtime application threads
81-89	IRQ threads
80	NFS
70-79	Soft IRQs
2-69	Regular applications
1	Low priority kernel threads

*/
        
#define _GNU_SOURCE
# include <stdio.h>
# include <stdlib.h>
# include <stdint.h>
# include <unistd.h>
# include <string.h>
# include <sys/time.h>
# include <sys/mman.h>
# include <limits.h>
# include <getopt.h>
# include <sched.h>
# include <sys/resource.h>
# include <errno.h>
# include <time.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <utmpx.h>
# include <asm/vsyscall.h>
# include <immintrin.h>

// gcc -W -Wall -O -avx2 -o HP-TimeTest7.2 HP-TimeTest7.2.c
// You'll need a recent version of gcc to use the -mtune=corei7-avx compiler flag
// This may require installing gmp-devel, and installing mpc and mpfr
// get gmp:  ./configure
//            make
//            make check
//            make install
// get mpfr: ./configure --with-gmp-include=/usr/local/include --with-gmp-lib=/usr/local/lib
//            make
//            make check
//            make install
// get mpc:   ./configure --with-gmp-include=/usr/local/include --with-gmp-lib=/usr/local/lib --with-mpfr-include=/usr/local/include --with-mpfr-lib=/usr/local/lib
//            make
//            make check
//            make install
// install the glibc-devel.i686 package
// get gcc:   ./configure --with-gmp-include=/usr/local/include --with-gmp-lib=/usr/local/lib --with-mpfr-include=/usr/local/include --with-mpfr-lib=/usr/local/lib --with-mpc-include=/usr/local/include --with-mpc-lib=/usr/local/lib
//            LD_LIBRARY_PATH=/usr/local/lib make
//            LD_LIBRARY_PATH=/usr/local/lib make check
//            LD_LIBRARY_PATH=/usr/local/lib make install

// LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64:/usr/local/lib /usr/local/bin/gcc -mavx -Wl,-Map=HP-TimeTest7.2-4.map,--cref -mtune=corei7-avx -march=corei7-avx -O HP-TimeTest7.2-4.c -o HP-TimeTest7.2-4
// LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64:/usr/local/lib /usr/local/bin/gcc -mavx -Wl,-Map=HP-TimeTest7.2-4.map,--cref -mtune=corei7-avx -march=corei7-avx -O HP-TimeTest7.2-4.c -S

/*
Edit history
Date		Version	Contact		Description
2011 02 03	6.1	Chuck Newman	Initial changed version, based on code from Channing Benson.  Changes include:
					> Prints the elapsed time since the program started
					> The threshold and loopcount are now optional arguments
					> Threshold is now the shortest time that gets reported vs. the longest time
					  that doesn't
					> Original version set its scheduling priority to the max and then locked pages.
					  It seems to me that opens a window wherein a page fault would be bad.
					  (i.e., I would think that paging can not happen at scheduling priority 99.)
					> Added a version number.  I received "timetest6.c" so I'm starting w/version 6.1
					> Added a "verbose" flag to increase or decrease the amount on stuff that's printed.
					> Added several options, including usage and an explanatory note.  Run w/ -h
2011 02 04	6.2	Chuck Newman	Print the minimum spike duration at verbosity level 2
					For this version (using gettimeofday()) it's pretty certain that there will be times
					in which consecutive calls return the same value (i.e., within the same usec
					resulting in a minimum spike of 0).
					However, I anticipate adding an option to use rdtsc, and in that case there will
					never be a minimum spike of 0.
2011 02 08	6.3	Chuck Newman	Add cycle-based jitter loop (in addition to existing time-based jitter loop).
					Also add associated usage information.
2011 04 25	?.?	Chuck Newman	Fix comment and help output; "--version" should be "--Version"
2011 08 18	6.4	Chuck Newman	If printing out a "spike" message, reacquire t_stamps[count%2] after fflush()
					This chage only needed for time-based loop; a comparable feature was already in
					the counter-based loop.
2012 03 02	6.5	Chuck Newman	If the verify setting was set to 0 then the scheduling priority would not be set
					This was fixed, and also the process is now set to be nice -20 (likely no effect).
2012 03 20	6.6	Chuck Newman	Store 1K spikes in a buffer and write them out when the buffer fills up.
					This is in contrast to the previous method in which I write them out every
					time they occur.
					Also add synthetic data and corresponding code to test changes (compile with -DFAKE)
					Pre-touch some stack and the memory where spike data is stored (may not be
					necessary because of the mlockall() call)
					Create some inline functions to simplify the code; that will also make it easier
					if I ever want to collapse the TIME and CYCLE loops into a single loop.
2012 03 23	7.0	Chuck Newman	Output can be printed in either CSV or XML as well as the previous freeform
					See the new "--format" parameter
2012 03 25	7.0	Chuck Newman	Units were incorrect in a couple of cases; didn't update version number
2012 04 19	7.1	Chuck Newman	I was printing out the month for the date directly from the "tm" structure
					where tm_mon is zero-based (e.g., April is 3).  Change to print it as
					one-based (e.g., April is 4).
2012 04 23	7.1	Chuck Newman	Included the option to gather and print out the MSR register SMI_COUNT
					For MSR_SMI_COUNT (34H, 52D), bits 31:0 are the SMI Count
					bits 63:32 are reserved.
					The information for reading the SMI count MSR came was designed by Naga Chumbalkar.
					For feedback, contact <nagac@hp.com>
					This doesn't build well on SLES 10.3, so I've moved to RHEL 5.3
					See the "Intel(R) 64 and IA-32 Architectures Software Developer's Manual, Volume 3B"
					at http://www.intel.com/Assets/en_US/PDF/manual/253669.pdf fields in the MSR
2012 04 23	7.1	Chuck Newman	If a latency spike happened within one microsecond (i.e., no change in the
					elapsed time) then the spike processing got confused; no bump in version.
2012 07 10	7.2	Chuck Newman	Add "--priority" option, giving user control over scheduler, priority, and nice.
					Renamed to HP-TimeTest
2013 07 18	7.2	Chuck Newman	Several months ago I included AVX instructions and a "power hog" option
					That requires a newer version of gcc, so today I added comments showing how to
					get the necessary components and build gcc, and how to compile with it.
2013 07 18	7.2	Chuck Newman	Print a header for the latency data.
2014 04 02	7.2	Chuck Newman	Missed "priority" in "long_options" structure; fixed.
2015 11 24	7.3	Chuck Newman	Add option to report overhead time or cycles, i.e., time or cycles passed while
					processing events.
					Fixed some formatting issues.

YYYY MM DD	V.V	F.N. L.N.	!!! UPDATE THE date_time VARIABLE BELOW AND THE Version VARIABLE BELOW !!!
*/

static char date_time[]="2015 11 24 20 35 UTC"; // YYYY MM DD HH MM
/* I want the date hand-coded in the source and not filled in by the compiler
static char date_time[]= __DATE__ " " __TIME__ ;
*/
typedef struct Version_struct {
   unsigned int major;
   unsigned int minor;
} Version_struct;
static Version_struct Version={7,3};

#define MAX_SPIKES 1021
typedef struct spike_data {
   unsigned int time;
   unsigned int spike;
} spike_data_struct;
/* Note that the array is dimensioned as MAX_SPIKES+3; this is to cover the unlikely case that
   the last spike is an extra-long one and therefore consumes three array pairs (described later).
*/
/* gcc 4.4.5 defines __BIGGEST_ALIGNMENT__; on the system I tested it on it came up as 16
   it was not defined with gcc 4.1.2, so I define it here if necessary.
*/
#ifndef __BIGGEST_ALIGNMENT__
#define __BIGGEST_ALIGNMENT__ 16
#endif
static spike_data_struct spikes[MAX_SPIKES+3] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
static char second_string[]="usec";
static char cycle_string[]="cycle";
static char *spike_unit;

#define TIME_METHOD   1
#define CYCLES_METHOD 2
#define method_default TIME_METHOD
#define threshold_time_default 10L
#define loopcount_time_default 5000000000L
#define threshold_cycles_default  10000L
#define loopcount_cycles_default  5000000000L

#define chatty_default 1
static unsigned int chatty=chatty_default;

#define CSV_FORMAT      1
#define XML_FORMAT      2
#define FREEFORM_FORMAT 4
static unsigned int format=FREEFORM_FORMAT;
static char XML_head[]="<!-- ";
static char XML_tail[]=" -->";
static int spike_header_printed=0;

#define DATE_OPTION		0
#define SMI_OPTION		1
#define POWER_HOG_OPTION	2
#define OVERHEAD_OPTION		3
#define LAST_OPTION		3
static int options[LAST_OPTION+1]={};

/* I couldn't find where these are specified in an include file or available through a system call. */
#define MAX_NICE  19
#define MIN_NICE -20

typedef struct timeval timesignature;

#ifdef FAKE
/* Synthetic data for testing.  Results should be:
  Run with "-v"
threshold=10 loopcount=28 verbosity=2
mlockall(): 0
sched_getscheduler(): SCHED_OTHER, 0
sched_get_priority_max():  99
sched_setscheduler(): 0
sched_getscheduler(): SCHED_FIFO, 99
getpriority(): 0
setpriority(): 0
getpriority(): -20
    0.000118 Latency spike of 99 usec
    0.000300 Latency spike of 145 usec
             182 usec since last spike
    1.000309 Latency spike of 1000007 usec
             1000009 usec since last spike
    2.000310 Latency spike of 1000000 usec
             1000001 usec since last spike
 4296.967632 Latency spike of 4294967295 usec
             4294967322 usec since last spike
 4300.000311 Latency spike of 3032677 usec
             3032679 usec since last spike

  Run with "-v -m cycles -t 10"
threshold=10 loopcount=24 verbosity=2
mlockall(): 0
sched_getscheduler(): SCHED_OTHER, 0
sched_get_priority_max():  99
sched_setscheduler(): 0
sched_getscheduler(): SCHED_FIFO, 99
getpriority(): 0
setpriority(): 0
getpriority(): -20
    0.000119 Latency spike of 99 usec
    0.000301 Latency spike of 145 usec
             182 usec since last spike
    1.000309 Latency spike of 1000007 usec
             1000008 usec since last spike
    2.000319 Latency spike of 1000000 usec
             1000010 usec since last spike
 4296.967633 Latency spike of 4294967295 usec
             4294967314 usec since last spike
 4300.000311 Latency spike of 3032677 usec
             3032678 usec since last spike

(N.B., the latency spikes will be the same for both cases, but the associated time may differ slightly.
 This is because for cycles the timeofday is only recorded when there is a spike in the cycle count
 whereas for times the timeofday value is used to determine spikes and therefore has already been recorded.)
*/

#define FAKE_SAMPLE_COUNT 32
#define FAKE_SPIKE_COUNT 4
static long high[FAKE_SAMPLE_COUNT]={1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1001, 1001, 1001, 1002, 1002, 1002, 1002,   5296,   5296,   5296, 5300, 5300, 5300};
static long  low[FAKE_SAMPLE_COUNT]={1000, 1001, 1009, 1010, 1018, 1019, 1118, 1119, 1119, 1128, 1136, 1137, 1145, 1146, 1154, 1155, 1300, 1301, 1302, 1309, 1309, 1310, 1310, 1319, 1328, 1337, 968632, 968633, 968634, 1311, 1311, 1312};
static int fake_data_ndx=0;
#endif

static inline void tt_gettime (timesignature* tvr) {
#ifndef FAKE
    int rv;
    rv=gettimeofday(tvr, NULL);
    if (rv!=0) { perror("Error calling gettimeofday"); fflush(stdout); fflush(stderr); }
#else
   tvr->tv_sec=(time_t)(high[fake_data_ndx]+(low[fake_data_ndx]>>32));
   tvr->tv_usec=(suseconds_t)(low[fake_data_ndx]&0xffffffffL);
   if (fake_data_ndx<FAKE_SAMPLE_COUNT-1) fake_data_ndx++;
#endif
   if (chatty >= 3) printf("%sGot a time of %5lu.%.6lu%s sec since the epoch began\n", XML_head, tvr->tv_sec, tvr->tv_usec, XML_tail);
}

static inline unsigned long get_cycles() {
#ifndef FAKE
    unsigned low, high;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    return ((unsigned long)high)<<32 | low;
#else
   unsigned long return_value=high[fake_data_ndx]*1000000L+low[fake_data_ndx];
   if (fake_data_ndx<FAKE_SAMPLE_COUNT-1) fake_data_ndx++;
   if (chatty >= 3) printf("%sGot a time of %5u %.6u%s sec since the epoch began\n", XML_head, return_value/1000000L, return_value-(return_value/1000000L)*1000000L, XML_tail);
   return return_value;
#endif
}

static inline unsigned long get_cycles_p() {
    unsigned low, high;
    asm volatile ("rdtscp\n\t"
       "mov %%edx, %0\n\t"
       "mov %%eax, %1\n\t":
       "=r"(high), "=r"(low)::"%rax", "%rcx", "%rdx");
    return low + ((unsigned long)(high)<<32);
}

static inline unsigned long tt_time_diff (timesignature* a, timesignature* b) {
    return (unsigned long)a->tv_sec * 1000000L + (unsigned long)a->tv_usec - ((unsigned long)b->tv_sec * 1000000L + (unsigned long)b->tv_usec);
}

static inline int compare_parameters(const char *str1, const char *gold) {
// -1 means different or test string is longer than the "gold" string
// non-negative means length of match before end of either string
   size_t loc=0, last;
   last=strlen(gold);
   if (strlen(str1)>last) return -1;
   if (strlen(str1)<last) last=strlen(str1);
   while (loc<last && (gold[loc]==str1[loc])) loc++;
   if (loc==last) return (int)loc;
   return -1;
}

static inline char *scheduler_string(int scheduler) {
   static char SCHED_FIFO_string[]  = "SCHED_FIFO";
   static char SCHED_RR_string[]    = "SCHED_RR";
   static char SCHED_OTHER_string[] = "SCHED_OTHER";
#ifdef SCHED_BATCH
   static char SCHED_BATCH_string[] = "SCHED_BATCH";
#endif
   static char unknown_string[]     = "unknown";

   switch (scheduler) {
      case SCHED_FIFO:
         return SCHED_FIFO_string;
      case SCHED_RR:
         return SCHED_RR_string;
      case SCHED_OTHER:
         return SCHED_OTHER_string;
#ifdef SCHED_BATCH
      case SCHED_BATCH:
         return SCHED_BATCH_string;
#endif
      default:
         return unknown_string;
   }
   return unknown_string;
}

static inline int parse_policy(char *policy_string, int *policy_value) {
   int rv_FIFO, rv_RR, rv_OTHER;
   int matches;
   rv_FIFO  = compare_parameters(policy_string, "FIFO");
   rv_RR    = compare_parameters(policy_string, "RR");
   rv_OTHER = compare_parameters(policy_string, "OTHER");
   matches=0;
   if (rv_FIFO>0)  { matches++; *policy_value=SCHED_FIFO;}
   if (rv_RR>0)    { matches++; *policy_value=SCHED_RR;}
   if (rv_OTHER>0) { matches++; *policy_value=SCHED_OTHER;}
   return matches;
}

static inline const char *policy_string(int policy) {
static const char FIFO_string[]="FIFO";
static const char RR_string[]="RR";
static const char OTHER_string[]="OTHER";
if (policy==SCHED_FIFO)  return FIFO_string;
if (policy==SCHED_RR)    return RR_string;
if (policy==SCHED_OTHER) return OTHER_string;
return NULL;
}

static inline int parse_priority(char *priority_string, int *priority_value) {
   int rv=0;
   char **endptr=NULL;
   *priority_value=strtol(priority_string, endptr, 10);
   if ( endptr != '\0' ) rv=1;
   return rv;
}

static inline int parse_nice(char *nice_string, int *nice_value) {
   int rv=0;
   char **endptr=NULL;
   *nice_value=strtol(nice_string, endptr, 10);
   if ( endptr != '\0' ) rv=1;
   return rv;
}

static inline void print_big_diff(spike_data_struct *spikes, unsigned int *spike_ndx) {
   static unsigned long cumulative=0L;
   unsigned long this_time;
   unsigned int ndx;
   if (chatty >= 3) printf("%sDump a buffer of up to %d spikes%s\n", XML_head, *spike_ndx, XML_tail);
   if (cumulative==0) {
      if ((chatty>0)&&(format==CSV_FORMAT)) printf("Elapsed Time (sec),spike (%s),delta time (usec)\n", spike_unit);
   }
   for (ndx=0; ndx<*spike_ndx; ndx++) {

/* look at both fields together in a single comparison */
      if ((*(unsigned long *)(&(spikes[ndx].time)))==0L) {
/* In this "if" section with both time and spike=0, the time consumes both "int"s of the next record
   and the spike consumes its normal field of the subsequent record;
   and the print statement has a corresponding long unsigned format.
*/
         this_time=*(unsigned long *)(&(spikes[ndx+1].time));
         cumulative+=this_time;
         if ( chatty>0 ) {
            if (format==FREEFORM_FORMAT) {
               printf("%5u.%.6u Latency spike of %u %s\n"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx+2].spike
                  , spike_unit);
               if (cumulative!=this_time) printf("             %lu usec since last spike\n"
                  , *(unsigned long *)(&(spikes[ndx+1].time)));
            } else if (format==CSV_FORMAT) {
               printf("%5u.%.6u,%u"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx+2].spike);
               if (cumulative!=this_time) printf(",%lu"
                  , *(unsigned long *)(&(spikes[ndx+1].time)));
               printf("\n");
            } else {
               printf("      <datum>\n         <elapsed>%u.%.6u</elapsed><spike>%u</spike>"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx+2].spike);
               if (cumulative!=this_time) printf("<delta>%lu</delta>"
                  , *(unsigned long *)(&(spikes[ndx+1].time)));
               printf("\n      </datum>\n");
            }
         }
         ndx+=2;
      } else {
         this_time=spikes[ndx].time;
         cumulative+=this_time;
         if ( chatty>0 ) {
            if (format==FREEFORM_FORMAT) {
               printf("%5u.%.6u Latency spike of %u %s\n"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx].spike
                  , spike_unit);
               if (cumulative!=this_time) printf("             %u usec since last spike\n"
                  , spikes[ndx].time);
            } else if (format==CSV_FORMAT) {
               printf("%5u.%.6u,%u"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx].spike);
               if (cumulative!=this_time) printf(",%u"
                  , spikes[ndx].time);
               printf("\n");
            } else {
               printf("      <datum>\n         <elapsed>%u.%.6u</elapsed><spike>%u</spike>"
                  , (unsigned int)(cumulative/1000000L)
                  , (unsigned int)(cumulative-(cumulative/1000000L)*1000000L)
                  , spikes[ndx].spike);
               if (cumulative!=this_time) printf("<delta>%u</delta>"
                  , spikes[ndx].time);
               printf("\n      </datum>\n");
            }
         }
      }
   }
   *spike_ndx=0;
   fflush( stdout );
}

static inline void process_big_diff(struct timeval *t_stamp, struct timeval *last_spike_time, spike_data_struct *spikes, unsigned int *spike_ndx, unsigned long diff) {
   unsigned long gap=(unsigned long)t_stamp->tv_sec * 1000000L + (unsigned long)t_stamp->tv_usec -
      (((unsigned long)last_spike_time->tv_sec * 1000000L) + (unsigned long)last_spike_time->tv_usec);
/* It's possible that there's a very long time between spikes (i.e., more than fits in a 32-bit counter).
   I would rather not allocate twice as much memory for those unlikely cases, so when that happens I set the time
   and spike values to 0 as a special case.  The next time-spike pair provides 64 bits for this long time,
   and the subsequent spike field is where I keep the spike value.
   
   It's possible that "gettimeofday" returns the same value for up to 1 microsecond of elapsed time,
   so it's conceivable that (for a very low threshold) a spike will happen within a single microsecond.
*/
   if (spike_header_printed == 0) {
      if (format==XML_FORMAT) {
         printf("%sElapsed time (seconds),latency spike (%s),delta time (%s)%s\n", XML_head, spike_unit, second_string, XML_tail);
      } else if (format==CSV_FORMAT) {
         printf("Elapsed time (seconds),latency spike (%s),delta time (%s)\n", spike_unit, second_string);
      }
      spike_header_printed = 1;
   }
   if (gap==(gap & 0xffffffffL)) {
      spikes[*spike_ndx].time=gap;
      spikes[*spike_ndx].spike=diff;
      if (chatty >= 3) printf("%sspikes[%d] = %6u %6u%s\n", XML_head, *spike_ndx, spikes[*spike_ndx].time, spikes[*spike_ndx].spike, XML_tail);
   } else {
      spikes[*spike_ndx].time=0;
      spikes[*spike_ndx].spike=0;
      *(unsigned long *)(&spikes[*spike_ndx+1].time)=gap;
      spikes[*spike_ndx+2].time=0xdeaddead;
      spikes[*spike_ndx+2].spike=diff;
      if (chatty >= 3) {
         printf("%sspikes[%d] = %6u %6u%s\n", XML_head, *spike_ndx, spikes[*spike_ndx].time, spikes[*spike_ndx].spike, XML_tail);
         printf("%sspikes[%d] = %13lu%s\n", XML_head, *spike_ndx, *(unsigned long *)(&spikes[*spike_ndx+1].time), XML_tail);
         printf("%sspikes[%d] = %6u %6u%s\n", XML_head, *spike_ndx, spikes[*spike_ndx+2].time, spikes[*spike_ndx+2].spike, XML_tail);
      }
      (*spike_ndx)+=2;
   }
   (*spike_ndx)++;
/* Filled up the buffer; time to print it.
*/
   if (*spike_ndx>=MAX_SPIKES) print_big_diff(spikes, spike_ndx);
   *last_spike_time = *t_stamp;
}

static int scheduler_priority() {
   int rv;
   int save_errno;
   struct sched_param sp;
   rv = sched_getparam(0, &sp);
   if (rv == -1) {
      save_errno = errno;
      if (chatty >= 1) printf ("%serror calling sched_getparam():  %s\nassuming scheduling priority is 0\n%s", XML_head, strerror(save_errno), XML_tail);
      return 0;
   } else {
      if (rv == 0) return sp.sched_priority;
   }
   fprintf(stderr, "sched_getparam() returned neither 0 nor -1; assuming scheduling priority is 0\n");
   return 0;
}

static int get_my_cpu() {
   int status;
   unsigned int core, node;
/* VSYSCALL_ADDR(__NR_vgetcpu) is not available on SLES 10.3
*/
   status=((int (*)(unsigned int *, unsigned int *, void *))(VSYSCALL_ADDR(__NR_vgetcpu)))(&core, &node, NULL);
   if (status != 0) { fprintf(stderr, "Error getting the current core number\n"); fflush (stderr); }
   return core;
}

static int msr_read(unsigned long MSR, unsigned long *value, unsigned long core __attribute__ ((__unused__)) ) {
   static int (*msr_fd_ptr)[]=NULL;
   static unsigned long (*SMI_count_ptr)[]=NULL;
   static long num_cores;
   int this_core;
   int count;
   off_t offset;
/* The "core" paramenter is not used, but the intent is to use it to get the MSR value for a different core
*/
/* Ensure we can access the output buffer
*/
   if (value==NULL) return -1;
   *value=0x8BadBeef;
   num_cores=sysconf(_SC_NPROCESSORS_CONF);
   if (num_cores == -1) {
      perror("unable to determine number of cores\n");
      options[SMI_OPTION]=0;
      return -1;
   }
   if (msr_fd_ptr == NULL) {
      msr_fd_ptr=(int (*)[])calloc(num_cores, sizeof(int));
      if (msr_fd_ptr == NULL) {
         perror("unable to allocate memory for SMI_COUNT functionality\n");
         options[SMI_OPTION]=0;
         return -1;
      }
      SMI_count_ptr=(unsigned long (*)[])calloc(num_cores, sizeof(unsigned long));
      if (SMI_count_ptr == NULL) {
         perror("unable to allocate memory for SMI_COUNT functionality\n");
         options[SMI_OPTION]=0;
         return -1;
      }
   }
   this_core = get_my_cpu();
/* sched_getcpu is not available on RHEL 5 variants
   this_core = sched_getcpu();
*/
   if (this_core<0) {
      perror("unable to determine current core\n");
      options[SMI_OPTION]=0;
      return -1;
   }
   if ((*msr_fd_ptr)[this_core] == 0) {
      char msr_path[64];
      sprintf(msr_path, "/dev/cpu/%u/msr", this_core);
      (*msr_fd_ptr)[this_core]=open(msr_path, O_RDONLY);
      if ((*msr_fd_ptr)[this_core]<0) {
         perror("unable to access /dev/cpu/<this core>/msr; perhaps the module is not loaded (try insmod msr)");
         options[SMI_OPTION]=0;
         return -1;
      }
   }
   offset = lseek((*msr_fd_ptr)[this_core], MSR, SEEK_SET);
   if (offset == -1) {
      perror("unable to access /dev/cpu/<this core>/msr\n");
      close((*msr_fd_ptr)[this_core]);
      options[SMI_OPTION]=0;
      return -1;
   }
   count=read((*msr_fd_ptr)[this_core], value, 8);
   if (count != 8) {
      perror("unable to access /dev/cpu/<this core>/msr\n");
      close((*msr_fd_ptr)[this_core]);
      options[SMI_OPTION]=0;
      return -1;
   }
   return count;
}

int main (const int argc, const char *const argv[])
{
   int ndx;
   int rv, status;
   int default_policy=SCHED_FIFO;
   int default_nice=-20;

   int requested_policy=SCHED_FIFO;
   int requested_priority=0, calculate_priority_flag=1;
   int requested_nice=-20;
   int priority_limit;
   int current_scheduler;
   unsigned long utempl;
   unsigned long count, diff, min_spike=ULONG_MAX;
   struct timeval t0_stamp;
   struct timeval last_spike_time = {0,0};
   struct timeval overhead_seconds={0L,0L};
   unsigned long overhead_cycles=0;

   int method=method_default;
   unsigned long threshold=0;
   unsigned long loopcount=0;
   int use_threshold_default=1;
   int use_loopcount_default=1;
   int option_index=0;
   unsigned int spike_ndx=0;

   unsigned long save_loopcount=0L;
   unsigned long save_threshold=0L;
   unsigned int save_chatty=0;
   int warm_up;

   struct option long_options[] = {
      {"method"   , required_argument, NULL, 'm'},
      {"threshold", required_argument, NULL, 't'},
      {"loopcount", required_argument, NULL, 'l'},
      {"format",    required_argument, NULL, 'f'},
      {"option",    required_argument, NULL, 'o'},
      {"priority",  required_argument, NULL, 'p'},
      {"Version",   no_argument,       NULL, 'V'},
      {"verbose",   optional_argument, NULL, 'v'},
      {"brief",     no_argument,       NULL, 'b'},
      {"explain",   no_argument,       NULL, 'e'},
      {"help",      no_argument,       NULL, 'h'},
      {NULL, 0, NULL, 0} };

/*
-v 3 -t 1000 -l 1000 asdf
*/
   while (optind<argc) {
/* this extra loop we're in now is to handle cases where an option is given an optional argument which is separated by a space.
   Currently the only option with an optional argument is "v" so an example would be "-v 3"
   Because I have "optstring" begin with a "+" processing will stop when an unknown option is encountered, e.g., "3"
   So if I find I've been interrupted before processing all options, I can examine the previously processed option.  If it's one
   that takes an optional argument and the unknown option "makes sense" as its argument, I'll treat it as such and restart option
   processing.
*/
      int last_rv='\0';
/* The "opt_optarg" variable is used to indicate that an option which can take an optional argument
   (e.g., v) did *NOT* in fact receive an optional
   argument.  If the next option that getopt() tries to process is not a valid option but can be
   interpreted as an argument to this option, then do so.  This means that, for example:
   -v2      is valid
   -v 3     is valid
   -v2 3    is invalid because "3" is not a valid argument to this program
*/
      int opt_optarg=0;
   while ( (rv=getopt_long (argc, (char *const *)argv, "+m:t:l:f:o:p:Vv::beh?", long_options, &option_index)) != -1 ) {
      int rv_cycles;
      int rv_time;
      int rv_csv, rv_xml, rv_freeform;
      int matches;
      last_rv=rv;
      switch (rv) {
         case 0:
            fprintf (stderr, "from getopt_long(), case 0 -- why are we here?\n");
            break;
         case 'm':
            rv_cycles = compare_parameters(optarg, "cycles");
            rv_time = compare_parameters(optarg, "time");
            if ( (rv_cycles>0) && (rv_time>0) ) {
               fprintf (stderr, "ambiguous value for method\n");
               exit (0);
            } else if ( (rv_cycles<0) && (rv_time<0) ) {
               fprintf (stderr, "illegal value for method; use \"cycles\" or \"time\"\n");
               exit (0);
            } else {
               if (rv_cycles>0) method=CYCLES_METHOD;
               else if (rv_time>0) method=TIME_METHOD;
               else {
                  fprintf (stderr, "value for method required; use \"cycles\" or \"time\"\n");
                  exit (0);
               }
            }
            break;
         case 't':
            utempl = strtoul(optarg, (char**) NULL, 10);
            if (utempl != 0) {
               threshold=utempl;
               use_threshold_default=0;
            }
            break;
         case 'l':
            utempl = strtoul(optarg, (char**) NULL, 10);
            if (utempl != 0) {
               loopcount=utempl;
               use_loopcount_default=0;
            }
            break;
         case 'f':
            rv_csv = compare_parameters(optarg, "csv");
            rv_xml = compare_parameters(optarg, "xml");
            rv_freeform = compare_parameters(optarg, "freeform");
            matches=0;
            if (rv_csv>0)      { matches++; format=CSV_FORMAT; };
            if (rv_xml>0)      { matches++; format=XML_FORMAT; };
            if (rv_freeform>0) { matches++; format=FREEFORM_FORMAT; };
            if (matches>1) {
               fprintf (stderr, "ambiguous value for format\n");
               exit (0);
            } else if (matches==0) {
               fprintf (stderr, "illegal value for format; use \"csv\" or \"xml\" or \"freeform\"\n");
               exit (0);
            }
            if (format!=XML_FORMAT) {
               XML_head[0]='\0';
               XML_tail[0]='\0';
            }
            break;
         case 'o':
            if (compare_parameters(optarg, "date") > 0) options[DATE_OPTION]=1;
            if (compare_parameters(optarg, "smi_count") > 0) options[SMI_OPTION]=1;
            if (compare_parameters(optarg, "overhead") > 0) options[OVERHEAD_OPTION]=1;
            if (compare_parameters(optarg, "power_hog") > 0) options[POWER_HOG_OPTION]=1;
            break;
         case 'p':
            if ( strlen(optarg) == 0L ) {
               fprintf (stderr, "illegal empty value for [priority][,policy][,nice]\n");
               exit (0);
            } else {
               char *optarg_copy, *policyp=NULL, *priorityp=NULL, *nicep=NULL;
               optarg_copy=strdup(optarg);
               if ( optarg_copy == NULL ) {
                  fprintf (stderr, "insufficient memory to process policy token\n");
                  exit (0);
               }
               if ( policyp=strsep(&optarg_copy, ",\0"),strlen(policyp) == 0 ) {
                  if (chatty >= 2) printf("%srequested default policy (%s)%s\n", XML_head, policy_string(default_policy), XML_tail);
                  requested_policy=default_policy;
               } else {
                  rv = parse_policy(policyp, &requested_policy);
                  if ( rv == 0 ) {
                     fprintf (stderr, "illegal value for policy; specify \"FIFO\" or \"RR\" or \"OTHER\"\n");
                     exit (0);
                  } else if ( rv > 1 ) {
                     fprintf (stderr, "ambiguous value for policy; specify \"FIFO\" or \"RR\" or \"OTHER\"\n");
                     exit (0);
                  }
                  if (chatty >= 2) printf("%srequested policy of %s%s\n", XML_head, policy_string(requested_policy), XML_tail);
               }

               if ( (optarg_copy == NULL) || (priorityp=strsep(&optarg_copy, ",\0"),strlen(priorityp) == 0) ) {
                  if (chatty >= 2) printf("%srequested default priority%s\n", XML_head, XML_tail);
                  calculate_priority_flag = 1;
               } else {
                  rv = parse_priority(priorityp, &requested_priority);
                  if ( rv != 0 ) {
                     fprintf (stderr, "illegal value for priority; specify an integer value\n");
                     exit (0);
                  }
                  calculate_priority_flag=0;
                  if (chatty >= 2) printf("%srequested priority of %d%s\n", XML_head, requested_priority, XML_tail);
               }

               if ( (optarg_copy == NULL) || (nicep=strsep(&optarg_copy, ",\0"),strlen(nicep) == 0) ) {
                  if (chatty >= 2) printf("%srequested default nice%s\n", XML_head, XML_tail);
                  requested_nice=default_nice;
               } else {
                  rv = parse_nice(nicep, &requested_nice);
                  if ( rv != 0 ) {
                     fprintf (stderr, "illegal value for nice; specify an integer value\n");
                     exit (0);
                  }
                  if (chatty >= 2) printf("%srequested nice of %d%s\n", XML_head, requested_nice, XML_tail);
                  if ( requested_nice > MAX_NICE ) {
                     if (chatty >= 1) printf("%srequested nice value of %d is too large; reducing to %d%s\n", XML_head, requested_nice, MAX_NICE, XML_tail);
                     requested_nice = MAX_NICE;
                  } else if ( requested_nice < MIN_NICE ) {
                     if (chatty >= 1) printf("%srequested nice value of %d is too small; increasing %d%s\n", XML_head, requested_nice, MIN_NICE, XML_tail);
                     requested_nice = MIN_NICE;
                  }
               }
            }
            break;
         case 'V':
            fprintf (stderr, "HP-TimeTest version %d.%d (%s)\n", Version.major, Version.minor, date_time);
            exit (0);
            break;
         case 'v':
            if ( optarg == NULL ) {
/* The "opt_optarg" variable is used to indicate that this option (v) did *NOT* receive an optional
   argument.  If the next option that getopt() tries to process is not a valid option but can be
   interpreted as an argument to this option, then do so.  This means that, for example:
   -v2      is valid
   -v 3     is valid
   -v2 3    is invalid
*/
               opt_optarg=0;
               chatty=2;
            } else {
               opt_optarg=1;
               utempl = strtoul(optarg, (char**) NULL, 10);
               if ( utempl > UINT_MAX ) utempl = UINT_MAX;
               chatty=(unsigned int)utempl;
            }
            break;
         case 'b':
            chatty=0;
            break;
         case 'e':
            printf ("This program locks its pages into memory and modifies its scheduling priority as\n"
                    "directed by the user (default is FIFO at 99) to lift it above interruptions.  It\n"
                    "then calls gettimeofday() repeatedly in a loop and tracks the amount of time\n"
                    "between these calls.  When the time difference is greater than the threshold (in\n"
                    "microseconds) then it prints a message, noting the time difference as a spike\n"
                    "and noting the length of time since the last latency spike.\n"
                    "\n"
                    "An alternate method is available, using cycles instead of microseconds.  The\n"
                    "rdtsc instruction is used instead of gettimeofday() and the threshold is\n"
                    "measured in cycles instead of microseconds.  This method is selected with the\n"
                    "\"--method=cycles\" option.\n"
                    "\n"
                    "You may want to specify values of time for arguments that take units of cycles.\n"
                    "For these cases you can convert based on the processor frequency.\n"
                    "E.g., if you want to use a threshold of 6 microseconds and run for 8 minutes\n"
                    "on a 2.7 GHz system, consider using the following:\n"
                    "\t-m cycles -t `echo '.000006 2700000000 * 0 k 1 / p' | dc` \\\n"
                    "\t-l `expr 2700000000 \\* 60 \\* 8 / 24`\n"
                    "The division by 24 corresponds to the number of cycles to perform one iteration\n"
                    "of the inner loop; you can find the corresponding number for your machine by\n"
                    "running a quick job with -m cycles -l 100 -v2\n"
                    "\n"
                    "It is presumed that these spikes are due to System Management Interrupts (SMIs).\n"
                    "Consider running this image on a selected core, but before doing so consider\n"
                    "precluding the Operating System from running software IRQs on that core.  The\n"
                    "following example illustrates one way of doing this for each core except core 0:\n"
                    "  until [ \"`service irqbalance status`\" = \"irqbalance is stopped\" ] ; do\n"
                    "    sleep 1 ; service irqbalance stop ; done\n"
                    "  Core=`grep -c processor /proc/cpuinfo` ; until [ $Core -eq 1 ] ; do\n"
                    "    Core=$(($Core-1))\n"
                    "    CoreMask=`echo \"16 o 2 $Core ^ p\" | dc`\n"
                    "    IRQBALANCE_ONESHOT=1 IRQBALANCE_BANNED_CPUS=${CoreMask} irqbalance\n"
                    "    echo Waiting for IRQ balancer to stop...\n"
                    "    until [ \"`service irqbalance status`\" = \"irqbalance is stopped\" ] ; do\n"
                    "      sleep 1 ; done\n"
                    "    echo \"--- Core $Core ---\"\n"
                    "    numactl --physcpubind=${Core} --localalloc nice -n -20 %s\n"
                    "  done\n"
                    , argv[0]);
         case 'h':
         case '?':
            printf ("usage:  [-m,  --method \"time\"|\"cycles\"(default=\"time\")]\n"
                    "        [-t,  --threshold #(default=%lu usecs|%lu cycles)]\n"
                    "        [-l,  --loopcount #(default=%lu (time)|%lu (cycles))]\n"
                    "        [-f,  --format \"csv\"|\"xml\"|\"freeform\"(default=freeform)]\n"
                    "        [-o,  --option \"date\" \"smi_count\" \"power_hog\" \"overhead\"]\n"
                    "        [-p,  --priority [\"FIFO\"|\"RR\"|\"OTHER\"(default policy=%s)][,#(default priority=sched_get_priority_max(=99 for FIFO,RR))][,#(default nice=%d)]\n"
                    "        [-V,  --Version]\n"
                    "        [-v#, --verbose=[#(default=%u]] [-b, --brief]\n"
                    "        [-e,  --explain] [-? -h, --help]\n",
               threshold_time_default, threshold_cycles_default, loopcount_time_default, loopcount_cycles_default, policy_string(default_policy), default_nice, chatty_default);
            exit (0);
            break;
         default:
            exit (0);
      }
   }
   if (optind!=argc) {
/* If we're here it's because we got an unrecognized option.  Figure out if it might be an argument
   to the previous option; if so treat it as such.
   Note that the getopt routines won't increment optind so we have to do it explicitly.
optarg
optind
opterr
optopt
*/      int valid_argument=0;
      if ((last_rv=='v')&&(opt_optarg==0)) {
         int scan_count;
         scan_count=sscanf(argv[optind], "%lu", &utempl);
//         utempl = strtoul(argv[optind], (char**) NULL, 10);
         if (scan_count != 1) break;
         valid_argument=1;
         opt_optarg=1;
         if ( utempl > UINT_MAX ) utempl = UINT_MAX;
         chatty=(unsigned int)utempl;
      }
      if (valid_argument==0) fprintf(stderr, "Unknown option %s\n", argv[optind]);
      optind++;
   }
   }

   time_t now1=time(NULL);
   struct tm *now2=localtime(&now1);
   if (format==CSV_FORMAT) {
      if ((chatty >= 2) || (options[DATE_OPTION] == 1))
         printf("Date and time,%d,%d,%d,%2.2d,%2.2d,%2.2d\n", now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
      if (chatty >= 2) {
         printf("Command");
         for (ndx=0; ndx<argc; ndx++)
            printf(",%s", argv[ndx]);
         printf("\n");
      }
      XML_head[0]='\0';
      XML_tail[0]='\0';
   } else if (format == XML_FORMAT) {
      printf("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n");
      if ((chatty >= 2) || (options[DATE_OPTION] == 1))
         printf("<date>\n  <year>%d</year>\n  <month>%d</month>\n  <day>%d</day>\n  <hour>%d</hour>\n  <minute>%d</minute>\n  <second>%d</second>\n</date>\n",
            now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
      if (chatty >= 2) {
         printf("<Command>");
         for (ndx=0; ndx<argc; ndx++)
            printf(" %s", argv[ndx]);
         printf("</Command>\n");
      }
   } else {
      if ((chatty >= 2) || (options[DATE_OPTION] == 1))
         printf("The current date and time are %d %d %d %2.2d:%2.2d:%2.2d\n", now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
      if (chatty >= 2) {
         printf("Command: ");
         for (ndx=0; ndx<argc; ndx++)
            printf(" %s", argv[ndx]);
         printf("\n");
      }
      XML_head[0]='\0';
      XML_tail[0]='\0';
   }
   if ( options[SMI_OPTION]==1) {
      unsigned long SMI_count;
      status=msr_read(0x34L, &SMI_count, 0L);
      if (status >= 0) {
         if (format == CSV_FORMAT) printf("SMI count,%ld\n", SMI_count);
         else if (format == XML_FORMAT) printf("<SMIcount>%ld</SMIcount>\n", SMI_count);
         else printf("SMI count:  %ld\n", SMI_count);
      }
   }

   if ( use_threshold_default == 1 ) {
      if (method == TIME_METHOD) threshold=threshold_time_default;
      if (method == CYCLES_METHOD) threshold=threshold_cycles_default;
   }
   if ( use_loopcount_default == 1 ) {
      if (method == TIME_METHOD) loopcount=loopcount_time_default;
      if (method == CYCLES_METHOD) loopcount=loopcount_cycles_default;
   }
   if (method == TIME_METHOD)
      spike_unit=second_string;
   else
      spike_unit=cycle_string;
#ifdef FAKE
   if (method == TIME_METHOD)
      loopcount=FAKE_SAMPLE_COUNT-FAKE_SPIKE_COUNT;
   else
      loopcount=FAKE_SAMPLE_COUNT-FAKE_SPIKE_COUNT*2;
#endif
   if (chatty >= 2) printf ("%sthreshold=%lu loopcount=%lu verbosity=%u%s\n", XML_head, threshold, loopcount, chatty, XML_tail);

/* Touch a bunch of memory we'll be needing.  It's my expectation that "stack" below will come from stack and not from heap. */
   {
      volatile long stack[MAX_SPIKES+3];
      unsigned int ndx=0;
      for (ndx=0;ndx<MAX_SPIKES;ndx++) {
         stack[ndx]=42L;
         spikes[ndx].time=42;
      }
      if ( stack[MAX_SPIKES+1] == 43 ) printf("We will never do this print\n");
   }

// The code originally had sched_setscheduler() before mlockall(); that seems backwards to me (Chuck Newman)
   rv = mlockall (MCL_CURRENT | MCL_FUTURE);
   if (chatty >= 1) printf ("%smlockall(): %d%s\n", XML_head, rv, XML_tail);

   if ( calculate_priority_flag == 1 ) {
/* Use default priority, which is the max available for the current scheduling policy */
      requested_priority = sched_get_priority_max(requested_policy);
   } else {
/* The user specified a priority; ensure that it is within the range of allowed values */
      priority_limit=sched_get_priority_min(requested_policy);
      if (requested_priority < priority_limit) {
         if (chatty >= 1) printf ("%srequested priority too low; increasing to minimum of %d%s\n", XML_head, priority_limit, XML_tail);
         requested_priority = priority_limit;
      } else {
         priority_limit=sched_get_priority_max(requested_policy);
         if (requested_priority > priority_limit) {
            if (chatty >= 1) printf ("%srequested priority too high; decreasing to maximum of %d%s\n", XML_head, priority_limit, XML_tail);
            requested_priority = priority_limit;
         }
      }
   }

   current_scheduler=sched_getscheduler(0);
   if (chatty >= 2) printf ("%ssched_getscheduler(): %s %d%s\n", XML_head, scheduler_string(current_scheduler), scheduler_priority(), XML_tail);
   struct sched_param sp = { requested_priority };
   rv = sched_setscheduler (0, requested_policy, &sp);
   if (chatty >= 1) printf ("%ssched_setscheduler(): %d%s\n", XML_head, rv, XML_tail);
   if (chatty >= 2) printf ("%ssched_getscheduler(): %s %d%s\n", XML_head, scheduler_string(sched_getscheduler(0)), scheduler_priority(), XML_tail);

   if (chatty >= 2) printf ("%sgetpriority(): %d%s\n", XML_head, getpriority(PRIO_PROCESS, 0), XML_tail);
   rv = setpriority (PRIO_PROCESS, 0, requested_nice);
   if (chatty >= 1) printf ("%ssetpriority(): %d%s\n", XML_head, rv, XML_tail);
   if (chatty >= 2) printf ("%sgetpriority(): %d%s\n", XML_head, getpriority(PRIO_PROCESS, 0), XML_tail);

   if (format==XML_FORMAT) {
      printf(
         "<spike_data>\n"
         "   <version>\n"
         "      <major>%d</major>\n"
         "      <minor>%d</minor>\n"
         "   </version>\n"
         "   <date>\n"
         "      <year>%d</year>\n"
         "      <month>%d</month>\n"
         "      <day>%d</day>\n"
         "      <hour>%2.2d</hour>\n"
         "      <minute>%2.2d</minute>\n"
         "      <second>%2.2d</second>\n"
         "   </date>\n"
         "   <source>\n"
         "      <program>\n"
         "         <name>HP-TimeTest</name>\n"
         "         <version>\n"
         "            <major>%d</major>\n"
         "            <minor>%d</minor>\n"
         "         </version>\n"
         "         <command>"
      , 1, 0, now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec, Version.major, Version.minor );
      {
         int ndx1;
         int pos, len, arglen;
         char *ptr;
         char space[2]={'\0','\0'};
         char delim;
         for(ndx1=0; ndx1<argc; ndx1++) {
            printf("%s", space);
            space[0]=' ';
            ptr=(char *)argv[ndx1];
            arglen=strlen(argv[ndx1]);
            pos=0;
            while (pos<arglen) {
               len=strcspn(ptr, "<>&'\"");
               delim=ptr[len];
               ptr[len]='\0';
               if (len>0) printf("%s", ptr);
               switch (delim) {
                  case '<':
                     printf("&lt");
                     break;
                  case '>':
                     printf("&gt");
                     break;
                  case '&':
                     printf("&amp");
                     break;
                  case '\'':
                     printf("&apos");
                     break;
                  case '"':
                     printf("&quot");
                     break;
                  case '\0':
                     break;
               }
               ptr+=len+1;
               pos+=len+1;
            }
         }
      }
      printf(
          "</command>\n"
          "      </program>\n"
          "   </source>\n"
          "   <data>\n"
          "      <field_1>\n"
          "         <name>elapsed</name>\n"
          "         <units>second</units>\n"
          "      </field_1>\n"
          "      <field_2>\n"
          "         <name>spike</name>\n"
          "         <units>%s</units>\n"
          "      </field_2>\n"
          "      <field_3>\n"
          "         <name>delta</name>\n"
          "         <units>usec</units>\n"
          "      </field_3>\n", spike_unit);
   }

   fflush( stdout ); fflush( stderr );

   warm_up=0;
   while ( warm_up++ < 2 ) {
      if ( warm_up == 1 ) {
         save_loopcount=loopcount;
         save_threshold=threshold;
         save_chatty=chatty;
         loopcount=MAX_SPIKES;
         threshold=0L;
         chatty=0L;
      } else {
         loopcount=save_loopcount;
         threshold=save_threshold;
         chatty=save_chatty;
         spike_ndx=0L;
         overhead_seconds.tv_sec=overhead_seconds.tv_usec=0L;
         overhead_cycles=0L;
      }
      tt_gettime (&t0_stamp);
      tt_time_diff(&t0_stamp,&t0_stamp);
      last_spike_time = t0_stamp;
      if (method==TIME_METHOD) {
         struct timeval t_stamps[2], temp_stamp;
         t_stamps[0]=t0_stamp;
         for (count = 1; count <= loopcount; count++) {
            tt_gettime (&t_stamps[count%2]);
            diff = tt_time_diff(&t_stamps[count%2], &t_stamps[(count-1)%2]);
            if (diff >= threshold) {
               process_big_diff(&(t_stamps[count%2]), &last_spike_time, spikes, &spike_ndx, diff);
               tt_gettime (&temp_stamp);
               overhead_seconds.tv_sec +=temp_stamp.tv_sec;
               overhead_seconds.tv_usec+=temp_stamp.tv_usec;
               overhead_seconds.tv_sec -=t_stamps[count%2].tv_sec;
/* Adjust for potential underflow and for overflow */
               if ( t_stamps[count%2].tv_usec > overhead_seconds.tv_usec ) {
                  overhead_seconds.tv_sec--;
                  overhead_seconds.tv_usec+=1000000;
               }
               overhead_seconds.tv_usec-=t_stamps[count%2].tv_usec;
               if (overhead_seconds.tv_usec >= 1000000) {
                  overhead_seconds.tv_sec++;
                  overhead_seconds.tv_usec-=1000000;
               }
               t_stamps[count%2]=temp_stamp;
            } else
               if (diff < min_spike) min_spike = diff;
         }
      } else {
         {
/* Get an initial value for min_spike */
            unsigned long cycle_stamp[2];
            cycle_stamp[0]=get_cycles_p();
            for (count = 1; count <= 1024; count++) {
               cycle_stamp[count%2]=get_cycles_p();
               diff = cycle_stamp[count%2] - cycle_stamp[(count-1)%2];
               if (diff < min_spike) min_spike = diff;
            }
         }
         if ( options[POWER_HOG_OPTION]==1) {
            unsigned long cycle_stamp[2];
//            double Array1[4*sizeof(__m256d)/sizeof(double)] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))={2.2360679774997896D};
//            double Array2[4*sizeof(__m256d)/sizeof(double)] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))={1.0D/2.2360679774997896D};
//            double Array3[16*sizeof(__m256d)/sizeof(double)] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))={2.2360679774997896D};
            volatile int never=0;
//            __m256d AVXymmA1, AVXymmA2, AVXymmA3, AVXymmA4;
//            __m256d AVXymmB1, AVXymmB2, AVXymmB3, AVXymmB4;
//            __m256d AVXymmC11, AVXymmC21, AVXymmC31, AVXymmC41;
//            __m256d AVXymmC12, AVXymmC22, AVXymmC32, AVXymmC42;
//            __m256d AVXymmC13, AVXymmC23, AVXymmC33, AVXymmC43;
//            __m256d AVXymmC14, AVXymmC24, AVXymmC34, AVXymmC44;
//            __m256d AVXymmT11, AVXymmT21, AVXymmT31, AVXymmT41;
//            __m256d AVXymmT12, AVXymmT22, AVXymmT32, AVXymmT42;
//            __m256d AVXymmT13, AVXymmT23, AVXymmT33, AVXymmT43;
//            __m256d AVXymmT14, AVXymmT24, AVXymmT34, AVXymmT44;
            cycle_stamp[0]=get_cycles();
//            AVXymmA1=_mm256_load_pd(&(Array1[0+never]));
//            AVXymmA2=_mm256_load_pd(&(Array1[4+never]));
//            AVXymmA3=_mm256_load_pd(&(Array1[8+never]));
//            AVXymmA4=_mm256_load_pd(&(Array1[12+never]));
//            AVXymmB1=_mm256_load_pd(&(Array2[0+never]));
//            AVXymmB2=_mm256_load_pd(&(Array2[4+never]));
//            AVXymmB3=_mm256_load_pd(&(Array2[8+never]));
//            AVXymmB4=_mm256_load_pd(&(Array2[12+never]));
//            AVXymmC11=_mm256_load_pd(&(Array3[0+never]));
//            AVXymmC12=_mm256_load_pd(&(Array3[4+never]));
//            AVXymmC13=_mm256_load_pd(&(Array3[8+never]));
//            AVXymmC14=_mm256_load_pd(&(Array3[12+never]));
//            AVXymmC21=_mm256_load_pd(&(Array3[16+never]));
//            AVXymmC22=_mm256_load_pd(&(Array3[20+never]));
//            AVXymmC23=_mm256_load_pd(&(Array3[24+never]));
//            AVXymmC24=_mm256_load_pd(&(Array3[28+never]));
//            AVXymmC31=_mm256_load_pd(&(Array3[32+never]));
//            AVXymmC32=_mm256_load_pd(&(Array3[36+never]));
//            AVXymmC33=_mm256_load_pd(&(Array3[40+never]));
//            AVXymmC34=_mm256_load_pd(&(Array3[44+never]));
//            AVXymmC41=_mm256_load_pd(&(Array3[48+never]));
//            AVXymmC42=_mm256_load_pd(&(Array3[52+never]));
//            AVXymmC43=_mm256_load_pd(&(Array3[56+never]));
//            AVXymmC44=_mm256_load_pd(&(Array3[60+never]));
            for (count = 1; count <= loopcount; count++) {
//               AVXymmT11=_mm256_mul_pd(AVXymmA1, AVXymmB1);
//               AVXymmT21=_mm256_mul_pd(AVXymmA2, AVXymmB1);
//               AVXymmT31=_mm256_mul_pd(AVXymmA3, AVXymmB1);
//               AVXymmT41=_mm256_mul_pd(AVXymmA4, AVXymmB1);
//               AVXymmT12=_mm256_mul_pd(AVXymmA1, AVXymmB2);
//               AVXymmT22=_mm256_mul_pd(AVXymmA2, AVXymmB2);
//               AVXymmT32=_mm256_mul_pd(AVXymmA3, AVXymmB2);
//               AVXymmT42=_mm256_mul_pd(AVXymmA4, AVXymmB2);
//               AVXymmT13=_mm256_mul_pd(AVXymmA1, AVXymmB3);
//               AVXymmT23=_mm256_mul_pd(AVXymmA2, AVXymmB3);
//               AVXymmT33=_mm256_mul_pd(AVXymmA3, AVXymmB3);
//               AVXymmT43=_mm256_mul_pd(AVXymmA4, AVXymmB3);
//               AVXymmT14=_mm256_mul_pd(AVXymmA1, AVXymmB4);
//               AVXymmT24=_mm256_mul_pd(AVXymmA2, AVXymmB4);
//               AVXymmT34=_mm256_mul_pd(AVXymmA3, AVXymmB4);
//               AVXymmT44=_mm256_mul_pd(AVXymmA4, AVXymmB4);
//               AVXymmC11=_mm256_add_pd(AVXymmC11, AVXymmT11);
//               AVXymmC21=_mm256_add_pd(AVXymmC21, AVXymmT21);
//               AVXymmC31=_mm256_add_pd(AVXymmC31, AVXymmT31);
//               AVXymmC41=_mm256_add_pd(AVXymmC41, AVXymmT41);
//               AVXymmC12=_mm256_add_pd(AVXymmC12, AVXymmT12);
//               AVXymmC22=_mm256_add_pd(AVXymmC22, AVXymmT22);
//               AVXymmC32=_mm256_add_pd(AVXymmC32, AVXymmT32);
//               AVXymmC42=_mm256_add_pd(AVXymmC42, AVXymmT42);
//               AVXymmC13=_mm256_add_pd(AVXymmC13, AVXymmT13);
//               AVXymmC23=_mm256_add_pd(AVXymmC23, AVXymmT23);
//               AVXymmC33=_mm256_add_pd(AVXymmC33, AVXymmT33);
//               AVXymmC43=_mm256_add_pd(AVXymmC43, AVXymmT43);
//               AVXymmC14=_mm256_add_pd(AVXymmC14, AVXymmT14);
//               AVXymmC24=_mm256_add_pd(AVXymmC24, AVXymmT24);
//               AVXymmC34=_mm256_add_pd(AVXymmC34, AVXymmT34);
//               AVXymmC44=_mm256_add_pd(AVXymmC44, AVXymmT44);
               cycle_stamp[count%2]=get_cycles();
               diff = cycle_stamp[count%2] - cycle_stamp[(count-1)%2];
               if (diff >= threshold) {
                  struct timeval spike_time;
                  tt_gettime(&spike_time);
                  process_big_diff(&spike_time, &last_spike_time, spikes, &spike_ndx, diff);
                  cycle_stamp[count%2]=get_cycles();
                  if ( never == 1 ) {
//                     AVXymmA1=_mm256_load_pd(&(Array2[count+0+never]));
//                     AVXymmA2=_mm256_load_pd(&(Array2[count+1+never]));
//                     AVXymmA3=_mm256_load_pd(&(Array2[count+2+never]));
//                     AVXymmA4=_mm256_load_pd(&(Array2[count+3+never]));
                  }
               } else
                  if (diff < min_spike) min_spike = diff;
            }
            if ( never == 1 ) {
//               _mm256_store_pd(&(Array3[0+never]), AVXymmC11);
//               _mm256_store_pd(&(Array3[4+never]), AVXymmC12);
//               _mm256_store_pd(&(Array3[8+never]), AVXymmC13);
//               _mm256_store_pd(&(Array3[12+never]), AVXymmC14);
//               _mm256_store_pd(&(Array3[16+never]), AVXymmC21);
//               _mm256_store_pd(&(Array3[20+never]), AVXymmC22);
//               _mm256_store_pd(&(Array3[24+never]), AVXymmC23);
//               _mm256_store_pd(&(Array3[28+never]), AVXymmC24);
//               _mm256_store_pd(&(Array3[32+never]), AVXymmC31);
//               _mm256_store_pd(&(Array3[36+never]), AVXymmC32);
//               _mm256_store_pd(&(Array3[40+never]), AVXymmC33);
//               _mm256_store_pd(&(Array3[44+never]), AVXymmC34);
//               _mm256_store_pd(&(Array3[48+never]), AVXymmC41);
//               _mm256_store_pd(&(Array3[52+never]), AVXymmC42);
//               _mm256_store_pd(&(Array3[56+never]), AVXymmC43);
//               _mm256_store_pd(&(Array3[60+never]), AVXymmC44);
//               printf("Never print the value %g\n", Array3[never]);
            }
         } else {
            unsigned long cycle_stamp[2], temp_cycles;
            struct timeval spike_time;
            cycle_stamp[0]=get_cycles_p();
            for (count = 1; count <= loopcount; count++) {
               cycle_stamp[count%2]=get_cycles_p();
               diff = cycle_stamp[count%2] - cycle_stamp[(count-1)%2];
               if (diff >= threshold) {
                  tt_gettime(&spike_time);
                  process_big_diff(&spike_time, &last_spike_time, spikes, &spike_ndx, diff);
                  temp_cycles=get_cycles_p();
                  overhead_cycles+=(temp_cycles-cycle_stamp[count%2]);
                  cycle_stamp[count%2]=temp_cycles;
               } else
                  if (diff < min_spike) min_spike = diff;
            }
         }
      }
   }
/* The full buffer has been dumped when it was filled;
   now that the loop is done the buffer has probably accumulated more spikes, so dump it.
*/
   if (spike_ndx>0) print_big_diff(spikes, &spike_ndx);

   if (min_spike != ULONG_MAX) {
// It is pretty much guaranteed that min_spike will be less than ULONG_MAX;
// if it is equal to ULONG_MAX then that means that every single iteration was a spike.
      if (chatty >= 2) {
         if (format == CSV_FORMAT) printf("minimum spike,%ld\n", min_spike);
         else if (format==XML_FORMAT) printf("      <minumum_spike>%lu</minumum_spike>\n", min_spike);
         else printf ("minimum spike = %lu units\n", min_spike);
      }
   }
   if (options[OVERHEAD_OPTION]==1) {
      if (method==TIME_METHOD) {
         if (format == CSV_FORMAT) printf("Overhead seconds,%lu.%.6lu\n", overhead_seconds.tv_sec, overhead_seconds.tv_usec);
         else if (format == XML_FORMAT) printf("<OverheadSeconds>%lu.%.6lu</OverheadSeconds>\n", overhead_seconds.tv_sec, overhead_seconds.tv_usec);
         else printf("Overhead seconds:  %lu.%.6lu\n", overhead_seconds.tv_sec, overhead_seconds.tv_usec);
      } else {
         if (format == CSV_FORMAT) printf("Overhead cycles,%ld\n", overhead_cycles);
         else if (format == XML_FORMAT) printf("      <OverheadCycles>%ld</OverheadCycles>\n", overhead_cycles);
         else printf("Overhead cycles = %ld\n", overhead_cycles);
      }
   }
   if (format==XML_FORMAT) {
      printf("   </data>\n</spike_data>\n");
   }
   if ( options[SMI_OPTION]==1) {
      unsigned long SMI_count;
      status=msr_read(0x34L, &SMI_count, 0L);
      if (status >= 0) {
         if (format == CSV_FORMAT) printf("SMI count,%ld\n", SMI_count);
         else if (format == XML_FORMAT) printf("<SMIcount>%ld</SMIcount>\n", SMI_count);
         else printf("SMI count:  %ld\n", SMI_count);
      }
   }
   if ((chatty >= 2) || (options[DATE_OPTION] == 1)) {
      time_t now1=time(NULL);
      struct tm *now2=localtime(&now1);
      if (format==CSV_FORMAT) printf("Date and time,%d,%d,%d,%2.2d,%2.2d,%2.2d\n", now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
      else if (format == XML_FORMAT) printf("<date>\n  <year>%d</year>\n  <month>%d</month>\n  <day>%d</day>\n  <hour>%d</hour>\n  <minute>%d</minute>\n  <second>%d</second>\n</date>\n",
         now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
      else printf("The current date and time are %d %d %d %2.2d:%2.2d:%2.2d\n", now2->tm_year+1900, now2->tm_mon+1, now2->tm_mday, now2->tm_hour, now2->tm_min, now2->tm_sec);
   }
   return 0;
}
